
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
import pandas as pd
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# |%%--%%| <boJ0kbIX53|DBR708U1GD>

# Transformations
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], is_check_shapes=False)

# |%%--%%| <DBR708U1GD|XFK99InvAY>

# Custom Dataset
class FloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.metadata.iloc[idx, 0])
        mask_name = os.path.join(self.mask_dir, self.metadata.iloc[idx, 1])
        
        image = np.array(Image.open(img_name).convert("RGB"))
        mask = np.array(Image.open(mask_name).convert("L"))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = mask / 255.0  # Normalize mask to [0, 1]
        return image, mask

# |%%--%%| <XFK99InvAY|ApOwNFf7uN>

# Data Loader
image_dir = 'Image'
mask_dir = 'Mask'
csv_file = 'metadata.csv'

dataset = FloodDataset(image_dir, mask_dir, csv_file, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# |%%--%%| <ApOwNFf7uN|Et6viCfdX6>

# Model
class SegmentationModel(nn.Module):
    def __init__(self, backbone):
        super(SegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Conv2d(backbone._fc.in_features, 1, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, x):
        features = self.backbone.extract_features(x)
        out = self.classifier(features)
        out = self.upsample(out)
        return out

# |%%--%%| <Et6viCfdX6|5rRyX2VgEF>

# Using a pretrained EfficientNet-B7 model
backbone = EfficientNet.from_pretrained('efficientnet-b3')
model = SegmentationModel(backbone)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

num_epochs = 20
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# |%%--%%| <5rRyX2VgEF|keP621yrV0>

def calculate_accuracy(outputs, masks):
    outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
    preds = outputs > 0.5  # Convert probabilities to binary predictions
    correct = (preds == masks).float()  # Compare predictions with masks
    accuracy = correct.sum() / correct.numel()  # Calculate accuracy
    return accuracy.item()

# |%%--%%| <keP621yrV0|Nh8MLqs9Jp>

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

# |%%--%%| <Nh8MLqs9Jp|INkygdRpbE>

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, masks)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader)
    print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
    
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        save_checkpoint(epoch, model, optimizer, epoch_loss, checkpoint_dir)
        # Ask user if they want to continue
        continue_training = input(f"Training reached epoch {epoch}. Do you want to continue? (yes/no): ")
        if continue_training.lower() != 'yes':
            print("Training stopped by user.")
            break

print("Training Complete!")
