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
from __future__ import print_function, division
import os
import math
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from metrics import smape
from model import Dual_efficientnet

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn, optim
from torchvision.transforms.functional import to_pil_image
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


import warnings

warnings.filterwarnings("ignore")


class StageDataset(Dataset):
    def __init__(self, excel_file, sheet_name, root_dir, data_info, transform=None):
        self.se_map_values = pd.read_excel(
            excel_file, sheet_name=sheet_name, 
        )  # Load the SE map values from an Excel file nrows=181
        self.root_dir = root_dir
        self.records = pd.read_excel(
            data_info, sheet_name=sheet_name, 
        )  # Load the records from an Excel file
        self.transform = transform
        self.image_slices = self.load_image_slices()  # Load the image slices

    def get_numeric_prefix(self, filename):
        numeric_prefix = ""
        for char in filename:
            if char.isdigit():
                numeric_prefix += char
            else:
                break
        return int(numeric_prefix)

    def load_image_slices(self):
        image_slices = []
        for idx in range(len(self.se_map_values)):
            img_folder_name = str(idx).zfill(
                4
            )  # Format the folder name with leading zeros
            img_folder = os.path.join(
                self.root_dir, img_folder_name
            )  # Get the path to the image folder
            print("Loading train images:", img_folder)
            img_files = glob.glob(
                os.path.join(img_folder, "*.jpg")
            )  # Get the list of image files
            img_files_sorted = sorted(
                img_files,
                key=lambda x: int(
                    os.path.splitext(os.path.basename(x))[0].split("_")[0]
                ),  # Sort the image files based on the numeric prefix in their names
            )

            images = []
            for i in range(0, len(img_files_sorted), len(img_files_sorted) // 128):
                img_file = img_files_sorted[i]
                image = Image.open(img_file)
                image = np.asarray(image) / 255.0
                image = np.transpose(image)
                image = image.astype(np.float32)
                images.append(image)
            image_slices.append(images)

        return image_slices

    def __len__(self):
        return len(self.se_map_values)

    def __getitem__(self, idx):
        image_slices = self.image_slices[idx]

        if self.transform:
            image_slices_pil = [
                to_pil_image(image_slice) for image_slice in image_slices
            ]
            image_slices_transformed = [
                self.transform(image_slice) for image_slice in image_slices_pil
            ]
            image_slices_transformed = torch.stack(image_slices_transformed).squeeze()
        else:
            image_slices_transformed = torch.tensor(image_slices)

        se_map_value = self.se_map_values.iloc[idx, 1:53].values.reshape(-1, 1)
        record = self.records.iloc[idx, 4:5].values

        if record[0] == "normal":
            record[0] = [0]
        elif record[0] == "early":
            record[0] = [1]
        elif record[0] == "intermediate":
            record[0] = [2]
        elif record[0] == "advanced":
            record[0] = [3]

        se_map_value = torch.from_numpy(se_map_value)
        record = torch.tensor(record[0])

        sample = {
            "image": image_slices_transformed,
            "record": record,
            "se_map_value": se_map_value,
        }
        return sample


class StageValDataset(Dataset):
    def __init__(self, excel_file, sheet_name, root_dir, data_info, transform=None):
        # total_rows = pd.read_excel(excel_file, sheet_name=sheet_name).shape[0]
        # skip_rows = total_rows - 10

        self.se_map_values = pd.read_excel(
            excel_file, sheet_name=sheet_name, skiprows=181
        )  # Load the SE map values from an Excel file, skipping the first 181 rows
        self.records = pd.read_excel(
            data_info, sheet_name=sheet_name, skiprows=181
        )  # Load the records from an Excel file, skipping the first 181 rows
        self.root_dir = root_dir
        self.transform = transform
        self.image_slices = self.load_image_slices()  # Load the image slices

    def load_image_slices(self):
        image_slices = []
        for idx in range(181, 201):  # Iterate over a specific range of indices
            img_folder_name = str(idx).zfill(
                4
            )  # Format the folder name with leading zeros
            img_folder = os.path.join(
                self.root_dir, img_folder_name
            )  # Get the path to the image folder
            print("Loading val images:", img_folder)
            img_files = glob.glob(
                os.path.join(img_folder, "*.jpg")
            )  # Get the list of image files
            img_files_sorted = sorted(
                img_files,
                key=lambda x: int(
                    os.path.splitext(os.path.basename(x))[0].split("_")[0]
                ),  # Sort the image files based on the numeric prefix in their names
            )

            images = []
            for i in range(0, len(img_files_sorted), len(img_files_sorted) // 128):
                img_file = img_files_sorted[i]
                image = Image.open(img_file)
                image = np.asarray(image) / 255.0
                image = np.transpose(image)
                image = image.astype(np.float32)
                images.append(image)
            image_slices.append(images)

        return image_slices

    def __len__(self):
        return len(self.se_map_values)

    def __getitem__(self, idx):
        image_slices = self.image_slices[idx]

        if self.transform:
            image_slices_pil = [
                to_pil_image(image_slice) for image_slice in image_slices
            ]
            image_slices_transformed = [
                self.transform(image_slice) for image_slice in image_slices_pil
            ]
            image_slices_transformed = torch.stack(image_slices_transformed).squeeze()
        else:
            image_slices_transformed = torch.tensor(image_slices)

        se_map_value = self.se_map_values.iloc[idx, 1:53].values.reshape(-1, 1)
        se_map_value = torch.from_numpy(se_map_value)

        record = self.records.iloc[idx, 4:5].values

        if record[0] == "normal":
            record[0] = [0]
        elif record[0] == "early":
            record[0] = [1]
        elif record[0] == "intermediate":
            record[0] = [2]
        elif record[0] == "advanced":
            record[0] = [3]
        record = torch.tensor(record[0])

        sample = {
            "image": image_slices_transformed,
            "record": record,
            "se_map_value": se_map_value,
        }

        return sample


# Parsing command-line arguments
parser = argparse.ArgumentParser(
    description="MICCAI2023 Challenge: STAGE, task 2 baseline training script"
)
parser.add_argument(
    "--excel-file",
    type=str,
    required=True,
    help="Path to the Excel file with annotations",
)
parser.add_argument(
    "--sheet-name",
    type=int,
    default=0,
    required=True,
    help="Name of the sheet containing the data",
)
parser.add_argument(
    "--root-dir", type=str, required=True, help="Directory with all the images"
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=0.0001,
    help="Learning rate for the optimizer",
)
parser.add_argument(
    "--num-epochs", type=int, default=20, help="Number of epochs for training"
)
parser.add_argument(
    "--batch-size", type=int, default=64, help="Batch size for training"
)
parser.add_argument(
    "--warmup-epochs", type=int, default=40, help="Number of warm-up epochs"
)
parser.add_argument(
    "--min-lr", type=float, default=0.00001, help="Minimum learning rate"
)

args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on the schedule"""
    if epoch < args.warmup_epochs:
        lr = args.learning_rate * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.learning_rate - args.min_lr) * 0.5 * (
            1
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.num_epochs - args.warmup_epochs)
            )
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# Data transforms
transform = transforms.Compose(
    [
        # transforms.Resize((512, 512)),
        transforms.CenterCrop((512, 512)),  # Add a central crop
        # transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

# Set device
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, prediction, label):
        mask = torch.ones_like(label)
        mask[label == 0] = 5
        loss = torch.mean(mask * (prediction - label) ** 2)
        return loss


loss_func = CustomMSELoss()

# loss_func = nn.MSELoss()

loss_func_name = "Mean squared error loss"

# Learning parameters
learn_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size

optimizer_name = "Adam"
start_time = datetime.now()

run_id = "checkpoint"
if os.path.exists(run_id):
    shutil.rmtree(run_id)
os.mkdir(run_id)

log_dir = run_id + "/logs"
writer = SummaryWriter(log_dir=log_dir)

# Write hyperparameters to a file
with open(run_id + "/hyperparams.csv", "w") as wfil:
    wfil.write("loss function, " + loss_func_name + "\n")
    wfil.write("learning rate (init), " + str(learn_rate) + "\n")
    wfil.write("number epochs, " + str(num_epochs) + "\n")
    wfil.write("batch size, " + str(batch_size) + "\n")
    wfil.write("optimizer, " + optimizer_name + "\n")
    wfil.write("start time, " + str(start_time) + "\n")

NUM_CLASSES = 52

# Create the Dual_network34 model
model = Dual_efficientnet()
print(model)

# Create the StageDataset for training
train_set = StageDataset(
    excel_file=args.excel_file,
    sheet_name=args.sheet_name,
    root_dir=args.root_dir,
    data_info="STAGE_training/data_info_training.xlsx",
    transform=transform,
)

# Create the StageValDataset for validation
valid_set = StageValDataset(
    excel_file=args.excel_file,
    sheet_name=args.sheet_name,
    root_dir=args.root_dir,
    data_info="STAGE_training/data_info_training.xlsx",
    transform=transform,
)

# Create data loaders
trainloader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=16
)
validloader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=16
)

# Freeze the first three layers of the model
ct = 0
for child in model.children():
    ct += 1
    if ct < 4:
        for param in child.parameters():
            param.require_grad = False

model.to(device)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.75, patience=3, verbose=True
)

best_valid_loss = "unset"
best_train_smape = float("inf")

# Log file for training progress
with open(run_id + "/log_file.csv", "w") as log_fil:
    log_fil.write("epoch,epoch duration,train loss,valid loss\n")

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        epoch_train_loss = 0.0
        epoch_valid_loss = 0.0
        epoch_train_smape = 0.0

        # Training
        model.train()
        for i, sample in tqdm(enumerate(trainloader), total=len(trainloader)):
            images = sample["image"]
            records = sample["record"]
            labels = sample["se_map_value"]

            images = images.to(device)
            records = records.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images, records)
            train_labels = torch.squeeze(labels)
            loss = loss_func(outputs, train_labels.float())
            smape_score = smape(
                train_labels.detach().cpu().numpy(), outputs.detach().cpu().numpy()
            )

            loss.backward()
            optimizer.step()

            epoch_train_loss = epoch_train_loss + loss.item()
            epoch_train_smape += smape_score

            iteration = epoch * len(trainloader) + i
            writer.add_scalar("Train Loss", loss.item(), iteration)
            writer.add_scalar(
                "Train SMAPE", epoch_train_smape / len(trainloader), epoch
            )

        # Validation
        model.eval()
        output_values = []
        with torch.no_grad():
            for i, sample in tqdm(enumerate(validloader), total=len(validloader)):
                images = sample["image"]
                records = sample["record"]
                labels = sample["se_map_value"]
                images = images.to(device)
                records = records.to(device)
                labels = labels.to(device)
                valid_labels = torch.squeeze(labels)
                outputs = model(images, records)
                loss = loss_func(outputs, valid_labels.float())
                epoch_valid_loss = epoch_valid_loss + loss.item()
                output_values.extend(outputs.cpu().tolist())

                # Calculate SMAPE
                y_true = valid_labels.cpu().numpy()
                y_pred = np.array(outputs.cpu())
                smape_score = smape(y_true, y_pred)
                print("Valid SMAPE:", smape_score)

        # Create a DataFrame from the output values
        df = pd.DataFrame(output_values)
        # Save the DataFrame to a CSV file
        df.to_csv("output_val.csv", index=False)

        adjust_learning_rate(optimizer, epoch, args)

        scheduler.step(epoch_valid_loss)

        writer.add_scalar(
            "Validation Loss", epoch_valid_loss / (NUM_CLASSES * batch_size), epoch
        )

        epoch_end = datetime.now()
        epoch_time = (epoch_end - epoch_start).total_seconds()

        # if best_valid_loss == "unset" or epoch_valid_loss < best_valid_loss:
        #     best_valid_loss = epoch_valid_loss
        #     print("Save best model!")
        #     torch.save(model.state_dict(), run_id + "/best_weights.pth")

        # torch.save(model.state_dict(), run_id + "/last_weights.pth")

        if epoch_train_smape / len(trainloader) < best_train_smape:
            best_train_smape = epoch_train_smape / len(trainloader)
            print("Save best model!")
            torch.save(model.state_dict(), run_id + "/best_weights.pth")

        torch.save(model.state_dict(), run_id + "/last_weights.pth")

        log_fil.write(
            str(epoch)
            + ","
            + str(epoch_time)
            + ","
            + str(epoch_train_loss / (NUM_CLASSES * batch_size))
            + ","
            + str(epoch_valid_loss / (NUM_CLASSES * batch_size))
            + "\n"
        )

        print(
            "epoch: "
            + str(epoch)
            + " - ("
            + str(epoch_time)
            + " seconds)"
            + "\n\ttrain loss: "
            + str(epoch_train_loss / (NUM_CLASSES * batch_size) / len(train_set))
            + "\n\tvalid loss: "
            + str(epoch_valid_loss / (NUM_CLASSES * batch_size) / len(valid_set))
        )

end_time = datetime.now()
with open(run_id + "/hyperparams.csv", "a") as wfil:
    wfil.write("end time," + str(end_time) + "\n")
