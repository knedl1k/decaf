#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50) -> None:
        super().__init__()
        self.s = s  # scale
        self.m = m  # margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        logits = F.linear(F.normalize(x), F.normalize(self.weight))

        if labels is None:
            return logits

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.m
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()

        logits = logits * self.s
        return logits


class MTGReconModel(nn.Module):
    def __init__(self, num_classes: int, embedding_size: int = 512) -> None:
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
        backbone_out = self.backbone.num_features

        self.bn1 = nn.BatchNorm1d(backbone_out)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(backbone_out, embedding_size, bias=False)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.arcface = ArcFaceHead(embedding_size, num_classes)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(x)

        features = self.bn1(features)
        features = self.dropout(features)
        features = self.fc(features)
        features = self.bn2(features)

        if labels is not None:
            return self.arcface(features, labels)
        return features
