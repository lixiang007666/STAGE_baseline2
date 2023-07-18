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
import os
import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms.functional import to_pil_image

from model import Dual_efficientnet


# Define a custom dataset class for the test dataset
class StageTestDataset(Dataset):
    def __init__(self, root_dir, sheet_name, data_info, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.records = pd.read_excel(data_info, sheet_name=sheet_name)
        self.image_slices = self.load_image_slices()

    def load_image_slices(self):
        image_slices = []
        # Iterate over the range of image folders (201 to 300)
        for idx in range(201, 301):
            img_folder_name = str(idx).zfill(4)
            img_folder = os.path.join(self.root_dir, img_folder_name)
            print(img_folder)
            # Get all image files in the folder
            img_files = glob.glob(os.path.join(img_folder, "*.jpg"))
            # Sort the image files based on the numerical prefix
            img_files_sorted = sorted(
                img_files,
                key=lambda x: int(
                    os.path.splitext(os.path.basename(x))[0].split("_")[0]
                ),
            )
            images = []
            # Load and preprocess the image slices
            for i in range(0, len(img_files_sorted), len(img_files_sorted) // 128):
                img_file = img_files_sorted[i]
                image = Image.open(img_file)
                image = np.asarray(image) / 255.0
                image = np.transpose(image)
                image = image.astype(np.float32)
                images.append(image)
            image_slices.append((img_folder, images))
        return image_slices

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        img_folder, image_slices = self.image_slices[idx]

        if self.transform:
            # Apply transformations to the image slices
            image_slices_pil = [
                to_pil_image(image_slice) for image_slice in image_slices
            ]
            image_slices_transformed = [
                self.transform(image_slice) for image_slice in image_slices_pil
            ]
            image_slices_transformed = torch.stack(image_slices_transformed).squeeze()
        else:
            image_slices_transformed = torch.tensor(image_slices)

        # Get the record value for the sample
        record = self.records.iloc[idx, 4:5].values.reshape(-1, 1)
        # Map the record value to numeric representation
        if record[0] == "normal":
            record[0] = 0
        elif record[0] == "early":
            record[0] = 1
        elif record[0] == "intermediate":
            record[0] = 2
        elif record[0] == "advanced":
            record[0] = 3
        record = torch.tensor(record[0].astype(np.float32))

        # Return the sample
        sample = {
            "img_folder": img_folder,
            "image": image_slices_transformed,
            "record": record,
        }
        return sample


# Set the device to CUDA if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify the path to the trained model weights
weight_file = "756/best_weights.pth"

# Set the batch size for testing
batch_size = 1

# Create an instance of the Dual_efficientnet model
model = Dual_efficientnet()

# Load the trained weights into the model
state_dict = torch.load(weight_file)
model.load_state_dict(state_dict)

# Set the model to evaluation mode and move it to the selected device
model.eval()
model.to(device)

# Define the transformation pipeline for the test dataset
transform = transforms.Compose(
    [
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
    ]
)

# Create an instance of the StageTestDataset for the test dataset
eval_set = StageTestDataset(
    root_dir="STAGE_validation/validation_images",
    sheet_name=0,
    data_info="STAGE_validation/data_info_validation.xlsx",
    transform=transform,
)

# Create a data loader for the test dataset
eval_loader = torch.utils.data.DataLoader(
    dataset=eval_set, batch_size=batch_size, shuffle=False
)

# Initialize lists to store the output values and image folder names
output_values = []
img_folders = []

# Iterate over the test dataset
for i, sample in enumerate(eval_loader):
    images = sample["image"]
    records = sample["record"]
    img_folder = sample["img_folder"]

    images = images.to(device)
    records = records.to(device)

    # Forward pass through the model
    outputs = model(images, records)

    updated_outputs = outputs.tolist()
    for i in range(len(updated_outputs[0])):
        if updated_outputs[0][i] < 1:
            updated_outputs[0][i] = 0

    # Append the output values and image folder names to the respective lists
    output_values.extend(updated_outputs)
    img_folders.extend(img_folder)

# Create a DataFrame from the output values
df = pd.DataFrame(output_values)
print(img_folders)
# Save the DataFrame to a CSV file
df.to_csv("output_test.csv", index=False)
