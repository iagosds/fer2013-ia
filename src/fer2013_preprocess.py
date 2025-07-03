import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define as transformações para as imagens
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Converte para escala de cinza
    transforms.RandomHorizontalFlip(), # Torna o modelo menos previsível, tornando-o mais acertivo em dados reais
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Carrega os datasets
train_dataset = datasets.ImageFolder(root='./fer2013/train',
                                     transform=transform_train)

val_dataset = datasets.ImageFolder(root='./fer2013/test',
                                   transform=transform_val)

# Cria os DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Número de classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")
print(f"Número de imagens de treino: {len(train_dataset)}")
print(f"Número de imagens de teste: {len(val_dataset)}")

# Salva os DataLoaders para uso posterior 
torch.save(train_loader, 'train_loader.pth')
torch.save(val_loader, 'val_loader.pth')
