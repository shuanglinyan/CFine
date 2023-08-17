# -*- coding: utf-8 -*-
"""

@author: zifyloo
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CRLoss(nn.Module):

    def __init__(self, opt):
        super(CRLoss, self).__init__()

        self.device = torch.device('cuda:{}'.format(opt.gpus))
        self.margin = np.array([opt.margin]).repeat(opt.batch_size)

    def semi_hard_negative(self, loss, margin):
        negative_index = np.where(np.logical_and(loss < margin, loss > 0))[0]
        return np.random.choice(negative_index) if len(negative_index) > 0 else None

    def hard_negative(self, loss, margin):
        negative_index = np.where(np.logical_and(loss > margin, loss > 0))[0]
        return np.argmax(loss) if len(negative_index) > 0 else None

    def get_triplets(self, similarity, labels, margin, semi):

        similarity = similarity.cpu().data.numpy()

        labels = labels.cpu().data.numpy()
        triplets = []

        for idx, label in enumerate(labels):  # same class calculate together
            negative = np.where(labels != label)[0]

            ap_sim = similarity[idx, idx]

            loss = similarity[idx, negative] - ap_sim + margin[idx]

            if semi:
                negetive_index = self.semi_hard_negative(loss, margin[idx])
            else:
                negetive_index = self.hard_negative(loss, margin[idx])

            if negetive_index is not None:
                triplets.append([idx, idx, negative[negetive_index]])

        if len(triplets) == 0:
            triplets.append([idx, idx, negative[0]])

        triplets = torch.LongTensor(np.array(triplets))

        return_margin = torch.FloatTensor(np.array(margin[triplets[:, 0]])).to(self.device)

        return triplets, return_margin

    def calculate_loss(self, similarity, label, margin, semi):

        image_triplets, img_margin = self.get_triplets(similarity, label, margin, semi)
        text_triplets, txt_margin = self.get_triplets(similarity.t(), label, margin, semi)

        image_anchor_loss = F.relu(img_margin
                                   - similarity[image_triplets[:, 0], image_triplets[:, 1]]
                                   + similarity[image_triplets[:, 0], image_triplets[:, 2]])

        similarity = similarity.t()
        text_anchor_loss = F.relu(txt_margin
                                  - similarity[text_triplets[:, 0], text_triplets[:, 1]]
                                  + similarity[text_triplets[:, 0], text_triplets[:, 2]])

        loss = torch.sum(image_anchor_loss) + torch.sum(text_anchor_loss)

        return loss

    def forward(self, similarity, labels, semi):

        cr_loss = self.calculate_loss(similarity, labels, self.margin, semi)
        return cr_loss

