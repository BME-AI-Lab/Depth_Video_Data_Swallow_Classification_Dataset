import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
from video_dataset import VideoFrameDataset, ImglistToTensor, NplistToTensor
import os
import torchvision.models.video as models

import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
from pathlib import Path

import tqdm

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import pytorchvideo.models.x3d


from sklearn.metrics import classification_report


from time import perf_counter


def make_x3d():
    return pytorchvideo.models.x3d.create_x3d(
        input_channel=3,  # RGB input from Kinetics
        input_clip_length=8,
        input_crop_size=224,  # For the tutorial let's just use a 50 layer network
        model_num_class=8,  # Kinetics has 400 classes so we need out final head to align
    )


if __name__ == "__main__":
    train_root = os.path.join(os.getcwd(), "train")
    train_annotation_file = os.path.join(train_root, "annotations.txt")

    val_root = os.path.join(os.getcwd(), "val")
    val_annotation_file = os.path.join(val_root, "annotations.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = transforms.Compose(
        [
            NplistToTensor(),
            # PackPathway(),
        ]
    )

    train_dataset = VideoFrameDataset(
        root_path=train_root,
        annotationfile_path=train_annotation_file,
        num_segments=4,
        frames_per_segment=2,
        imagefile_template="{:04d}.npy",
        transform=preprocess,
        test_mode=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = VideoFrameDataset(
        root_path=val_root,
        annotationfile_path=val_annotation_file,
        num_segments=4,
        frames_per_segment=2,
        imagefile_template="{:04d}.npy",
        transform=preprocess,
        test_mode=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = make_x3d().to(device)

    num_epochs = 200
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=1e-4
    )  # learning rate, optimizer SGD / adam

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/SlowFast_trainer_{}".format(timestamp))
    epoch_number = 0

    torch.backends.cudnn.benchmark = True

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        batch_num = 0

        for video_batch, labels in tqdm.tqdm(train_dataloader):
            # Every data instance is an input + label pair2
            # video_batch = [j.to(device, non_blocking=True) for j in video_batch]
            video_batch = video_batch.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)

            # Zero your gradients for every batch!
            optimizer.zero_grad(set_to_none=True)

            # Make predictions for this batch
            outputs = model(video_batch)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            # running_loss += loss.item()
            running_loss += loss.item()  # loss per batch
            if batch_num % 10 == 9:
                last_loss = running_loss / 10  # loss per batch
                # print("  batch {} loss: {}".format(batch_num + 1, last_loss))
                tb_x = epoch_index * len(train_dataloader) + batch_num + 1
                writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0
            batch_num += 1
        # running_loss = 0.

        return last_loss

    best_vloss = 1_000_000.0
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("EPOCH {}:".format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)
        model.train(False)
        running_vloss = 0.0
        i = 0
        running_accuracy = 0
        total = 0
        alllabels = list()
        allpreds = list()
        with torch.no_grad():
            for vvideo_batch, vlabels in tqdm.tqdm(val_dataloader):
                vvideo_batch = vvideo_batch.to(device, non_blocking=True)
                vlabels = vlabels.to(device, non_blocking=True)

                voutputs = model(vvideo_batch)
                _, predicted = torch.max(voutputs, 1)
                total += vlabels.size(0)
                running_accuracy += (predicted == vlabels).sum().item()

                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss
                i += 1
                alllabels = alllabels + vlabels.cpu().tolist()
                allpreds = allpreds + predicted.cpu().tolist()
        avg_vloss = running_vloss / (i + 1)
        report = classification_report(alllabels, allpreds, output_dict=True)
        print(
            "LOSS train {} valid {}, F1-score valid {}".format(
                avg_loss, avg_vloss, report["accuracy"]
            )
        )

        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        if avg_vloss < best_vloss or report["accuracy"] > best_acc:
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
            if report["accuracy"] > best_acc:
                best_acc = report["accuracy"]
            model_path = "model_path".format(
                timestamp, epoch_number
            )
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"trained_model_epoch{epoch+1}.pt")
