import matplotlib.pyplot as plt
import os
import cv2
import torch
import numpy as np
# visualize loss & accuracy
def visualize_curve(log_root):
    log_file = open(log_root, 'r')
    result_root = log_root[:log_root.rfind('/') + 1] + 'train.jpg'
    loss = []
    
    top1_i2t = []
    top10_i2t = []
    top1_t2i = []
    top10_t2i = []
    for line in log_file.readlines():
        line = line.strip().split()
        
        if 'top10_t2i' not in line[-2]:
            continue
        
        loss.append(line[1])
        top1_i2t.append(line[3])
        top10_i2t.append(line[5])
        top1_t2i.append(line[7])
        top10_t2i.append(line[9])

    log_file.close()

    plt.figure('loss')
    plt.plot(loss)

    plt.figure('accuracy')
    plt.subplot(211)
    plt.plot(top1_i2t, label = 'top1')
    plt.plot(top10_i2t, label = 'top10')
    plt.legend(['image to text'], loc = 'upper right')
    plt.subplot(212)
    plt.plot(top1_t2i, label = 'top1')
    plt.plot(top10_i2t, label = 'top10')
    plt.legend(['text to image'], loc = 'upper right')
    plt.savefig(result_root)
    plt.show()
def visualize_img(attn_mat,img):
    attn_mat=attn_mat.detach().numpy()
    mask=cv2.resize(attn_mat,224,224)
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    plt.show()
def visualize_image(att_mat, im):
    # att_mat = torch.stack(att_mat).squeeze(1)
    #
    # # Average the attention weights across all heads.
    # att_mat = torch.mean(att_mat, dim=2)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    # residual_att = torch.eye(att_mat.size(1)).cpu()
    # aug_att_mat = att_mat + residual_att
    # aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #
    # # Recursively multiply the weight matrices
    # joint_attentions = torch.zeros(aug_att_mat.size())
    # joint_attentions[0] = aug_att_mat[0]
    #
    # for n in range(1, aug_att_mat.size(0)):
    #     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = att_mat
    #print(v.shape)
    grid_size = 14
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    return mask