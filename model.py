"""
Copyright 2023 The HDMI Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pdb

import torch
import torch.nn as nn
import torchvision.models as models

from efficientnet_pytorch import EfficientNet


class Dual_efficientnet(nn.Module):
    def __init__(self):
        super(Dual_efficientnet, self).__init__()

        self.oct_branch = EfficientNet.from_pretrained("efficientnet-b5")

        # Replace first conv layer in oct_branch
        self.oct_branch._conv_stem = nn.Conv2d(
            128, 48, kernel_size=3, stride=2, padding=3, bias=False
        )

        self.embedding_layer = nn.Embedding(4, 512)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.oct_branch._fc = nn.Sequential()

        input_size = 2048 + 512

        self.regression_head = nn.Linear(input_size, 52)

        # Batch Normalization layer
        self.batch_norm = nn.BatchNorm2d(2048)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, oct_img, record):
        embedded_category_features = self.embedding_layer(record.long())

        b = self.oct_branch.extract_features(oct_img)

        # Apply Batch Normalization
        b = self.gap(self.batch_norm(b)).squeeze(dim=2).squeeze(dim=2)

        regression_input = torch.cat([b, embedded_category_features.squeeze(dim=1)], 1)

        # Apply Dropout
        regression_input = self.dropout(regression_input)

        regression_output = self.regression_head(regression_input)

        return regression_output
