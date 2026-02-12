import torch
import torch.nn as nn
import torchvision.models as models
from model import MobileViT
import numpy as np

class SwinMobileViTHybrid(nn.Module):
    """
    Hybrid model combining Swin Transformer and MobileViT for feature extraction.
    Features are concatenated and used with gradient boosting classifiers.
    """
    def __init__(self, num_classes=8, swin_variant='base', mobilevit_weights_path=None, freeze_backbones=True):
        super(SwinMobileViTHybrid, self).__init__()
        
        self.num_classes = num_classes
        self.swin_variant = swin_variant
        
        # Initialize Swin Transformer
        if swin_variant == 'base':
            self.swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            self.swin_feature_dim = 1024  # Swin-B output dimension
        elif swin_variant == 'small':
            self.swin = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
            self.swin_feature_dim = 768  # Swin-S output dimension
        else:
            raise ValueError(f"Invalid swin_variant: {swin_variant}. Choose 'base' or 'small'.")
        
        # Remove the classification head from Swin
        self.swin.head = nn.Identity()
        
        # Initialize MobileViT
        image_size = 224
        dims = [96, 192, 384]
        channels = [16, 32, 64, 128, 160, 192, 256, 320, 384, 512]
        depths = [2, 2, 2]  # Using best configuration from optimization
        
        self.mobilevit = MobileViT(
            image_size=image_size,
            num_classes=num_classes,
            dims=dims,
            channels=channels,
            depths=depths
        )
        
        # Load pre-trained MobileViT weights if provided
        if mobilevit_weights_path:
            print(f"Loading MobileViT weights from {mobilevit_weights_path}")
            self.mobilevit.load_state_dict(torch.load(mobilevit_weights_path, map_location='cpu'))
        
        # Remove the classification head from MobileViT
        self.mobilevit_feature_dim = dims[-1]  # 384
        self.mobilevit.fc = nn.Identity()
        
        # Freeze backbones if specified
        if freeze_backbones:
            for param in self.swin.parameters():
                param.requires_grad = False
            for param in self.mobilevit.parameters():
                param.requires_grad = False
        
        # Combined feature dimension
        self.combined_feature_dim = self.swin_feature_dim + self.mobilevit_feature_dim
        
        # Optional: Add a fusion layer (can be disabled for pure feature extraction)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.final_feature_dim = 256
        
        # Temporary classifier for end-to-end training (will be replaced by gradient boosting)
        self.temp_classifier = nn.Linear(self.final_feature_dim, num_classes)
    
    def extract_features(self, x):
        """
        Extract features from both Swin and MobileViT backbones.
        Returns concatenated features.
        """
        # Extract Swin features
        swin_features = self.swin(x)  # (B, swin_feature_dim)
        
        # Extract MobileViT features
        mobilevit_features = self.mobilevit(x)  # (B, mobilevit_feature_dim)
        
        # Concatenate features
        combined_features = torch.cat([swin_features, mobilevit_features], dim=1)  # (B, combined_feature_dim)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)  # (B, final_feature_dim)
        
        return fused_features
    
    def forward(self, x):
        """
        Forward pass for end-to-end training.
        For gradient boosting, use extract_features() instead.
        """
        features = self.extract_features(x)
        output = self.temp_classifier(features)
        return output
    
    def get_feature_dim(self):
        """Returns the dimension of extracted features."""
        return self.final_feature_dim


def extract_features_from_dataset(model, dataloader, device):
    """
    Extract features from entire dataset using the hybrid model.
    Returns features and labels as numpy arrays for gradient boosting training.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model.extract_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels


if __name__ == '__main__':
    # Test the hybrid model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test with Swin-Base
    print("\n=== Testing Swin-Base + MobileViT ===")
    model_base = SwinMobileViTHybrid(num_classes=8, swin_variant='base', freeze_backbones=True)
    model_base = model_base.to(device)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    features = model_base.extract_features(dummy_input)
    output = model_base(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature dimension: {model_base.get_feature_dim()}")
    
    # Test with Swin-Small
    print("\n=== Testing Swin-Small + MobileViT ===")
    model_small = SwinMobileViTHybrid(num_classes=8, swin_variant='small', freeze_backbones=True)
    model_small = model_small.to(device)
    
    features_small = model_small.extract_features(dummy_input)
    print(f"Extracted features shape (Swin-Small): {features_small.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_base.parameters())
    trainable_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
