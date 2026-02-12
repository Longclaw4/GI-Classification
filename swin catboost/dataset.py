import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class KvasirDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
    ])

    # Create dataset instance
    dataset_root = 'kvasir-dataset-v2/kvasir-dataset-v2'
    kvasir_dataset = KvasirDataset(root_dir=dataset_root, transform=transform)

    # Create DataLoader instance
    batch_size = 32
    dataloader = DataLoader(kvasir_dataset, batch_size=batch_size, shuffle=True)

    # Example usage: Iterate through a batch of data
    for images, labels in dataloader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break

