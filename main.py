from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import torch
import torchvision.models as models
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# virtual env: cv
def main():
    # vit
    # model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    # resnet18
    model = models.resnet18()
    model.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

    dataset_root = './output_folder/train'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    # 过滤器函数，只选择图像文件
    def is_valid_file(file_name):
        return file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    dataset = ImageFolder(root=dataset_root, transform=transform, is_valid_file=is_valid_file)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("length of validation set is ",len(val_dataloader))
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # validate the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            count = 0
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # hahaha
                count += (labels == 1).sum().item()
            accuracy = correct / total
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}, count: {count}')
            
if __name__ == '__main__':
    main()