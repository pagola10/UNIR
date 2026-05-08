import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt

train_df = pd.read_csv('sign_mnist_train.csv')

labels = train_df['label'].values
pixels = train_df.drop(columns='label').values
images = pixels.reshape(-1, 1, 28, 28).astype(np.float32)

sample_idx = 0
sample_tensor = torch.tensor(images[sample_idx] / 255.0)

transform_dict = {
    "Original": T.ToPILImage(),
    "Rotación": T.Compose([T.ToPILImage(), T.RandomRotation(30)]),
    "Traslación": T.Compose([T.ToPILImage(), T.RandomAffine(0, translate=(0.2, 0.2))]),
    "Escalado": T.Compose([T.ToPILImage(), T.RandomAffine(0, scale=(0.7, 1.3))]),
    "Inversión": T.Compose([T.ToPILImage(), T.RandomInvert(p=1.0)]),
    "Ruido": T.Compose([T.ToPILImage(), T.ToTensor(), lambda x: x + torch.randn_like(x) * 0.1]),
    "Recorte": T.Compose([T.ToPILImage(), T.RandomResizedCrop(28, scale=(0.6, 1.0))])
}

plt.figure(figsize=(15, 5))
for i, (name, trans) in enumerate(transform_dict.items()):
    plt.subplot(1, 7, i+1)
    img_display = trans(sample_tensor)
    if not isinstance(img_display, torch.Tensor):
        plt.imshow(img_display, cmap='gray')
    else:
        plt.imshow(img_display.squeeze(), cmap='gray')
    plt.title(name)
    plt.axis('off')
plt.tight_layout()
plt.show()

X_tensor = torch.tensor(images / 255.0)
y_tensor = torch.tensor(labels, dtype=torch.long)

full_dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

class SignLanguageCNN(nn.Module):
    def __init__(self, k_size=3, st=1, pad=1, p_size=2):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=k_size, stride=st, padding=pad)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=p_size, stride=p_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=k_size, stride=st, padding=pad)
        
        self.flat_size = self._calc_flat_size(p_size)
        self.fc = nn.Linear(self.flat_size, 25)

    def _calc_flat_size(self, p_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            x = self.pool(self.relu(self.conv1(dummy)))
            x = self.pool(self.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SignLanguageCNN(k_size=3, st=1, pad=1, p_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    t_loss, t_acc = 0, 0
    for X, y in train_loader:
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_acc += (out.argmax(1) == y).sum().item()
    
    model.eval()
    v_loss, v_acc = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            out = model(X)
            loss = criterion(out, y)
            v_loss += loss.item()
            v_acc += (out.argmax(1) == y).sum().item()
            
    print(f"Epoch {epoch+1} | Train Loss: {t_loss/len(train_loader):.4f} | Train Acc: {100*t_acc/train_size:.2f}%")
    print(f"Epoch {epoch+1} | Val Loss: {v_loss/len(val_loader):.4f} | Val Acc: {100*v_acc/val_size:.2f}%")
