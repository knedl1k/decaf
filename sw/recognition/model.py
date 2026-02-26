#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # scale (gradient increase)
        self.m = m  # margin (class separation)

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # normalization of input vectors and weights to length 1 (Hypersphere)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if labels is None:
            # during inference, we only need the cosine similarity
            return cosine

        # trimming values so they don't overflow the acos function
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)

        # adding margin to the angle of the correct class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        target_logit = torch.cos(theta + self.m)
        output = one_hot * target_logit + (1.0 - one_hot) * cosine
        output *= self.s

        return output


class MTGReconModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(MTGReconModel, self).__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        backbone_out = self.backbone.num_features
        self.bn1 = nn.BatchNorm1d(backbone_out)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(backbone_out, embedding_size, bias=False)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.arcface = ArcFaceHead(embedding_size, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)

        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc(features)
        features = self.bn2(features)

        if labels is not None:
            return self.arcface(features, labels)

        return features
