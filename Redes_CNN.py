import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

train_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(p=0.5),           # 1. Flip horizontal
    transforms.RandomRotation(degrees=30),             # 2. Rotación
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 3. Brillo y contraste
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)), # 4. Recorte aleatorio
    transforms.RandomGrayscale(p=0.1),                # 5. Escala de grises parcial
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_path_train = './images/seg_train/seg_train'
data_path_test = './images/seg_test/seg_test'

train_dataset = ImageFolder(root=data_path_train, transform=train_transforms)
test_dataset = ImageFolder(root=data_path_test, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

def imshow(img):
    img = img / 2 + 0.5     # Desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)
print("Visualización de un lote de entrenamiento (Con transformaciones):")
imshow(torchvision.utils.make_grid(images))

class IntelCNN(nn.Module):
    def __init__(self):
        super(IntelCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        
        self.fc1 = nn.Linear(32 * 36 * 36, 128)
        self.fc2 = nn.Linear(128, 6) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 36 * 36) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = IntelCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

epochs = 30
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch + 1} - Loss: {running_loss / len(train_loader):.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Precisión final en imágenes de prueba: {100 * correct / total:.2f}%')
