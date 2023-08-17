import os
import sys
import time
import shutil
import logging
import gc
import torch
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk
from test_config import config
from config import data_config, network_config, get_image_unique
import numpy as np
import math
import re
from matplotlib import pyplot as plt
from datasets.pedes import CuhkPedes
from utils.visualize import visualize_image, visualize_img


def test(data_loader, network, args, unique_image, epoch=0):
    batch_time = AverageMeter()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    # switch to evaluate mode
    network.eval()
    max_size = 6156
    img_feat_bank = torch.zeros(args.num_heads + 1, max_size, args.feature_size)
    text_feat_bank = torch.zeros(args.num_heads + 1, max_size, args.feature_size)
    labels_bank = torch.zeros(max_size)
    index = 0
    with torch.no_grad():
        end = time.time()
        for images, captions, labels in data_loader:
            (
                tokens,
                segments,
                input_masks,
                caption_length,
            ) = network.module.language_model.pre_process(captions)

            tokens = tokens.cuda()
            segments = segments.cuda()
            input_masks = input_masks.cuda()
            images = images.cuda()
            labels = labels.cuda()
            interval = images.shape[0]    # 64

            img_output, text_output = network(
                images, tokens, segments, input_masks
            )

            for i in range(len(img_output)):
                img_feat_bank[i][index : index + interval] = img_output[i]
                text_feat_bank[i][index : index + interval] = text_output[i]
            labels_bank[index : index + interval] = labels
            batch_time.update(time.time() - end)
            end = time.time()
            index = index + interval
        unique_image = torch.tensor(unique_image) == 1

        result, score = compute_topk(
            img_feat_bank[:, unique_image, :],
            text_feat_bank,
            labels_bank[unique_image],
            labels_bank,
            [1, 5, 10],
            True,
        )
        (
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
        ) = result
        return (
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
            batch_time.avg,
        )


def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    test_loader = data_config(
        args.image_dir, args.anno_dir, 64, "test", 100, test_transform
    )
    unique_image = get_image_unique(
        args.image_dir, args.anno_dir, 64, "test", 100, test_transform
    )

    logging.info("Testing on dataset: {}".format(args.anno_dir))
    
    model_file = os.path.join(args.model_path, "best_model.pth.tar")
    network, _ = network_config(args, "test", None, True, model_file)

    (
        ac_top1_i2t,
        ac_top5_i2t,
        ac_top10_i2t,
        ac_top1_t2i,
        ac_top5_t2i,
        ac_top10_t2i,
        test_time,
    ) = test(test_loader, network, args, unique_image)

    logging.info(
        "top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}".format(
            ac_top1_t2i,
            ac_top5_t2i,
            ac_top10_t2i,
            ac_top1_i2t,
            ac_top5_i2t,
            ac_top10_i2t,
        )
    )
    logging.info(args.model_path)
    logging.info(args.log_dir)


if __name__ == "__main__":
    args = config()
    main(args)
