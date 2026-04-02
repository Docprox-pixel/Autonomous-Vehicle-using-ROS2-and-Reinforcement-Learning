import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np

class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        # NVIDIA Architecture (5 CNNs + 3 Dense)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 34, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1) # Steering prediction (regression)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class SteeringDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file, header=None)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = image / 255.0 # Normalize
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        steering = self.data.iloc[idx, 1].astype(np.float32)
        return torch.from_numpy(image), torch.tensor([steering])

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PilotNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dataset = SteeringDataset('dataset/labels.csv', 'dataset/images')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f'Starting training on {device}...')
    for epoch in range(10): # 10 Epochs for prototype
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}')
        
    torch.save(model.state_dict(), 'pilotnet.pth')
    print('Model saved to pilotnet.pth')

if __name__ == '__main__':
    train()
