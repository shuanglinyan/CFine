import torch.nn as nn
from .resnet import resnet50
from .bert import Bert
import torchvision.models as models
import torch
import os
import cv2
import math
import clip
import copy
import torch.nn.functional as F
import pickle
import numpy as np
from .modeling import VisionTransformer, CONFIGS
from models.clip import VisionTransformer_clip
from models.layers import TransformerDecoder, ResidualAttention

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ViT_CLIP_custom(nn.Module):
    def __init__(self, model_name='ViT-B/16', device=None):
        super(ViT_CLIP_custom, self).__init__()
        self.device = device
        clip_model, _ = clip.load(model_name, device=self.device)
        state_dict = clip_model.visual.state_dict()
        vision_width = state_dict["conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("transformer.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["conv1.weight"].shape[-1]
        grid_size = round((state_dict["positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
        vision_heads = vision_width // 64
        self.clip_model = VisionTransformer_clip(input_resolution=image_resolution,
                                            patch_size=vision_patch_size,
                                            width=vision_width,
                                            layers=vision_layers,
                                            heads=vision_heads)
        if "proj" in state_dict:
            del state_dict["proj"]
        self.clip_model.load_state_dict(state_dict)

    def forward(self, x):

        x, att_score = self.clip_model(x)
        return x, att_score


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        nn.init.constant_(m.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.image_model_name = args.img_model
        self.img_k_ratio = args.img_k_ratio   # 0.1
        self.text_k_ratio = args.text_k_ratio # 0.2
        self.kw = args.pos_kw                 # 3
        self.kp = args.pos_kp                 # 3
        self.scale_cs = args.scale_cs         # 0.07
        self.scale_cd = args.scale_cd         # 0.01

        if args.img_model == "vit":
            config = CONFIGS[args.model_type]
            self.image_model = VisionTransformer(config)
            self.image_model.load_from(np.load(args.pretrain_dir))
        elif args.img_model == "clip":
            self.image_model = ViT_CLIP_custom()
            self.image_model.clip_model.float()
        if args.text_model == "bert":
            self.language_model = Bert()

        self.block = ResidualAttention(num_layers=1,
                                       d_model=768,
                                       n_head=12,
                                       att_type='cross')

        self.bottleneck_image = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)
        self.bottleneck_text = nn.BatchNorm1d(args.feature_size)
        self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)
        self.similarityNorm = nn.Softmax(dim=2)

    def forward(self, images, tokens, segments, input_masks):
        if self.image_model_name == 'clip':
            image_feats, image_att_scores = self.image_model(images)
        else:
            image_feats = self.image_model(images)[0]
        text_feats, text_att_scores = self.language_model(tokens, segments, input_masks)
        # global
        img_global = self.bottleneck_image(image_feats[:, 0, :])
        text_global = self.bottleneck_text(text_feats[:, 0, :])

        text_masks = torch.zeros_like(tokens).masked_fill_(tokens == 0, 1).bool()

        # fine-grain
        image_parts = []
        text_parts = []
        img_score = image_att_scores[-1][:, 0, :]
        text_score = text_att_scores[-1][:, 0, :]
        temp = torch.zeros(text_score.size(0), 1)
        score_mask_img = (torch.cat((temp, torch.ones(img_score.size(0), img_score.size(1)-1)), dim=1)).to(img_score.device)
        score_mask_text = (torch.cat((temp, torch.ones(text_score.size(0), text_score.size(1) - 1)), dim=1)).to(text_score.device)
        img_score = img_score * score_mask_img
        text_score = text_score * score_mask_text

        image_feats_selected1 = []
        image_feats_selected2 = []
        image_feats_selected_all = []
        img_K = int(image_feats.size(1) * self.img_k_ratio)
        for b in range(image_feats.size(0)):
            _, idx = img_score[b].topk(img_score.size(1), largest=True, sorted=True)
            neg_idx1 = idx[:img_K]
            neg_idx2 = idx[img_K:img_K*2]
            image_feats_selected1.append(image_feats[b][neg_idx1, :])
            image_feats_selected2.append(image_feats[b][neg_idx2, :])
            image_feats_selected_all.append(image_feats[b][idx[:img_K*2], :])
        image_feats_selected1 = torch.stack(image_feats_selected1, dim=0)
        image_feats_selected1 = torch.cat((image_feats[:, 0, :].unsqueeze(1), image_feats_selected1), dim=1)
        image_feats_selected2 = torch.stack(image_feats_selected2, dim=0)
        image_feats_selected2 = torch.cat((image_feats[:, 0, :].unsqueeze(1), image_feats_selected2), dim=1)
        image_feats_selected_all = torch.stack(image_feats_selected_all, dim=0)

        # ResidualAttentionCross
        image_part1 = self.block(image_feats_selected1, image_feats)
        image_part2 = self.block(image_feats_selected2, image_feats)

        image_parts.append(image_part1)
        image_parts.append(image_part2)

        text_feats_selected1 = []
        text_mask_selected1 = []
        text_feats_selected2 = []
        text_mask_selected2 = []
        text_feats_selected_all = []
        text_K = int(text_feats.size(1) * self.text_k_ratio)
        for b in range(text_feats.size(0)):
            _, idx = text_score[b].topk(text_score.size(1), largest=True, sorted=True)
            neg_idx1 = idx[:text_K]
            neg_idx2 = idx[text_K:text_K*2]
            text_feats_selected1.append(text_feats[b][neg_idx1, :])
            text_mask_selected1.append(text_masks[b][neg_idx1])
            text_feats_selected2.append(text_feats[b][neg_idx2, :])
            text_mask_selected2.append(text_masks[b][neg_idx2])
            text_feats_selected_all.append(text_feats[b][idx[:text_K*2], :])
        text_feats_selected1 = torch.stack(text_feats_selected1, dim=0)
        text_mask_selected1 = torch.stack(text_mask_selected1, dim=0)
        text_feats_selected1 = torch.cat((text_feats[:, 0, :].unsqueeze(1), text_feats_selected1), dim=1)
        text_mask_selected1 = torch.cat((text_masks[:, 0].unsqueeze(1), text_mask_selected1), dim=1)
        text_feats_selected2 = torch.stack(text_feats_selected2, dim=0)
        text_mask_selected2 = torch.stack(text_mask_selected2, dim=0)
        text_feats_selected2 = torch.cat((text_feats[:, 0, :].unsqueeze(1), text_feats_selected2), dim=1)
        text_mask_selected2 = torch.cat((text_masks[:, 0].unsqueeze(1), text_mask_selected2), dim=1)
        text_feats_selected_all = torch.stack(text_feats_selected_all, dim=0)

        # ResidualAttentionCross
        text_part1 = self.block(text_feats_selected1, text_feats, text_masks)
        text_part2 = self.block(text_feats_selected2, text_feats, text_masks)

        text_parts.append(text_part1)
        text_parts.append(text_part2)

        if self.training:
            # ---------------------------
            # Cross-Similarity
            # ---------------------------
            G_img_token = image_feats[:, 0, :].unsqueeze(1)
            L_img_token = image_feats_selected_all
            B = L_img_token.size(0)
            G_text_token = text_feats[:, 0, :].unsqueeze(1)
            L_text_token = text_feats_selected_all

            G_img_token_norm = G_img_token / G_img_token.norm(dim=-1, keepdim=True)
            L_img_token_norm = L_img_token / L_img_token.norm(dim=-1, keepdim=True)
            G_text_token_norm = G_text_token / G_text_token.norm(dim=-1, keepdim=True)
            L_text_token_norm = L_text_token / L_text_token.norm(dim=-1, keepdim=True)

            # image-word sim
            G_img_token_norm_l = G_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
            L_text_token_norm_r = L_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)

            sim_iw = torch.matmul(G_img_token_norm_l, L_text_token_norm_r.transpose(-2, -1)) / self.scale_cs
            weight_iw = F.softmax(sim_iw, dim=-1)
            sim_iw = torch.mul(sim_iw, weight_iw)
            sim_iw = torch.sum(sim_iw, dim=-1).squeeze()

            # piexl-text sim
            L_img_token_norm_l = L_img_token_norm.unsqueeze(1).repeat(1, B, 1, 1)
            G_text_token_norm_r = G_text_token_norm.unsqueeze(0).repeat(B, 1, 1, 1)

            sim_pt = torch.matmul(L_img_token_norm_l, G_text_token_norm_r.transpose(-2, -1)) / self.scale_cs
            weight_pt = F.softmax(sim_pt, dim=2)
            sim_pt = torch.mul(sim_pt, weight_pt)
            sim_pt = torch.sum(sim_pt, dim=2).squeeze()

            sim_cs = (sim_iw + sim_pt) / 2

            # ---------------------------
            # Correspondence Discovery
            # ---------------------------
            L_img_token1 = image_feats_selected1[:, 1:, :]
            L_text_token1 = text_feats_selected1[:, 1:, :]
            L_img_token_norm1 = L_img_token1 / L_img_token1.norm(dim=-1, keepdim=True)
            L_text_token_norm1 = L_text_token1 / L_text_token1.norm(dim=-1, keepdim=True)
            _b, _n, _c = L_text_token1.shape
            vidwordSim = torch.bmm(L_img_token_norm1, L_text_token_norm1.permute(0, 2, 1))
            vidwordSim = self.similarityNorm(vidwordSim)
            # ---------- posWord -------
            _, idxWord = vidwordSim.topk(self.kw, dim=2, largest=True, sorted=True)
            posWord = []
            for _batch in range(idxWord.shape[0]):
                posWord.append(L_text_token1[_batch, idxWord[_batch], :])
            posWord = torch.stack(posWord)
            posWord = torch.mean(posWord, dim=2)
            posWord_norm = posWord / posWord.norm(dim=-1, keepdim=True)
            # ---------- posClip -------
            _, idxVid = vidwordSim.topk(self.kp, dim=1, largest=True, sorted=True)
            posClip = []
            for _batch in range(idxVid.shape[0]):
                posClip.append(L_img_token1[_batch, idxVid[_batch], :])
            posClip = torch.stack(posClip)
            posClip = torch.mean(posClip, dim=1)
            posClip_norm = posClip / posClip.norm(dim=-1, keepdim=True)

            # piexl_word sim
            L_img_token_norm_l1 = L_img_token_norm1.unsqueeze(1).repeat(1, _b, 1, 1)
            posWord_norm_r = posWord_norm.unsqueeze(0).repeat(_b, 1, 1, 1)
            posClip_norm_l = posClip_norm.unsqueeze(1).repeat(1, _b, 1, 1)
            L_text_token_norm_r1 = L_text_token_norm1.unsqueeze(0).repeat(_b, 1, 1, 1)

            sim_pw0 = torch.matmul(L_img_token_norm_l1, posWord_norm_r.transpose(-2, -1)) / self.scale_cd
            sim_pw0 = torch.diagonal(sim_pw0, dim1=-2, dim2=-1)
            sim_pw0 = torch.mean(sim_pw0, dim=2)

            sim_pw1 = torch.matmul(posClip_norm_l, L_text_token_norm_r1.transpose(-2, -1)) / self.scale_cd
            sim_pw1 = torch.diagonal(sim_pw1, dim1=-2, dim2=-1)
            sim_pw1 = torch.mean(sim_pw1, dim=2)

            sim_cd = (sim_pw0 + sim_pw1) / 2

        img_output = (img_global,)
        text_output = (text_global,)

        for j in range(len(image_parts)):
            img_output = img_output + (self.bottleneck_image(image_parts[j][:, 0, :]),)
            text_output = text_output + (self.bottleneck_text(text_parts[j][:, 0, :]),)
        img_f = torch.stack(img_output, dim=1)
        text_f = torch.stack(text_output, dim=1)
        if self.training:
            return img_output, text_output, img_f, text_f, sim_cs, sim_cd
        else:
            return img_output, text_output
