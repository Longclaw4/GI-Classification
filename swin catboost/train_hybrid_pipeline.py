"""
Hybrid Model Training Pipeline: Swin Transformer + MobileViT + Gradient Boosting
This script trains the complete hybrid architecture with Optuna optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import time
import os
from tqdm import tqdm

from dataset import KvasirDataset
from hybrid_model import SwinMobileViTHybrid, extract_features_from_dataset
from gradient_boosting_classifier import GradientBoostingClassifier

def main():
    # Configuration
    config = {
        'dataset_root': 'kvasir-dataset-v2/kvasir-dataset-v2',
        'mobilevit_weights': 'mobilevit_kvasir_v2_best_optuna.pth',  # Pre-trained MobileViT
        'swin_variant': 'small',  # 'base' or 'small'
        'num_classes': 8,
        'batch_size': 32,
        'image_size': 224,
        'freeze_backbones': True,
        'classifier_types': ['xgboost', 'lightgbm', 'catboost'],
        'optuna_sampler': 'tpe',  # 'tpe' or 'bohb'
        'optuna_trials': 50,
        'output_dir': 'hybrid_results'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"HYBRID MODEL TRAINING PIPELINE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Swin Variant: {config['swin_variant']}")
    print(f"Freeze Backbones: {config['freeze_backbones']}")
    print(f"Optuna Sampler: {config['optuna_sampler'].upper()}")
    print(f"{'='*70}\n")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = KvasirDataset(root_dir=config['dataset_root'], transform=transform)
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create dataloaders
    pin_memory = True if torch.cuda.is_available() else False
    num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,  # No need to shuffle for feature extraction
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # Initialize hybrid model
    print(f"\nInitializing Swin-{config['swin_variant'].upper()} + MobileViT hybrid model...")
    hybrid_model = SwinMobileViTHybrid(
        num_classes=config['num_classes'],
        swin_variant=config['swin_variant'],
        mobilevit_weights_path=config['mobilevit_weights'],
        freeze_backbones=config['freeze_backbones']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in hybrid_model.parameters())
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Feature dimension: {hybrid_model.get_feature_dim()}")
    
    # Extract features from all datasets
    print("\n" + "="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    
    print("\nExtracting training features...")
    start_time = time.time()
    X_train, y_train = extract_features_from_dataset(hybrid_model, train_loader, device)
    train_time = time.time() - start_time
    print(f"Training features: {X_train.shape}, Time: {train_time:.2f}s")
    
    print("Extracting validation features...")
    start_time = time.time()
    X_val, y_val = extract_features_from_dataset(hybrid_model, val_loader, device)
    val_time = time.time() - start_time
    print(f"Validation features: {X_val.shape}, Time: {val_time:.2f}s")
    
    print("Extracting test features...")
    start_time = time.time()
    X_test, y_test = extract_features_from_dataset(hybrid_model, test_loader, device)
    test_time = time.time() - start_time
    print(f"Test features: {X_test.shape}, Time: {test_time:.2f}s")
    
    # Save extracted features
    features_path = os.path.join(config['output_dir'], 'extracted_features.npz')
    np.savez(
        features_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"\nFeatures saved to {features_path}")
    
    # Train gradient boosting classifiers
    print("\n" + "="*70)
    print("GRADIENT BOOSTING CLASSIFIER TRAINING")
    print("="*70)
    
    results = {}
    
    for classifier_type in config['classifier_types']:
        print(f"\n{'#'*70}")
        print(f"# Training {classifier_type.upper()}")
        print(f"{'#'*70}\n")
        
        # Initialize classifier
        gb_classifier = GradientBoostingClassifier(
            classifier_type=classifier_type,
            num_classes=config['num_classes']
        )
        
        # Optimize hyperparameters
        study = gb_classifier.optimize(
            X_train, y_train, 
            X_val, y_val,
            n_trials=config['optuna_trials'],
            sampler=config['optuna_sampler']
        )
        
        # Train with best parameters
        gb_classifier.train_with_best_params(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        metrics = gb_classifier.evaluate(X_test, y_test)
        
        # Save model
        model_path = os.path.join(config['output_dir'], f'{classifier_type}_model')
        gb_classifier.save(model_path)
        
        # Store results
        results[classifier_type] = {
            'best_params': gb_classifier.best_params,
            'metrics': metrics,
            'best_val_accuracy': study.best_value
        }
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(f"\n{'Classifier':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUROC':<12}")
    print("-"*70)
    
    for classifier_type, result in results.items():
        metrics = result['metrics']
        print(f"{classifier_type.upper():<15} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['auroc']:<12.4f}")
    
    # Find best classifier
    best_classifier = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
    print("\n" + "="*70)
    print(f"BEST CLASSIFIER: {best_classifier[0].upper()}")
    print(f"Test Accuracy: {best_classifier[1]['metrics']['accuracy']:.4f}")
    print(f"Test AUROC: {best_classifier[1]['metrics']['auroc']:.4f}")
    print("="*70)
    
    # Save summary
    import json
    summary_path = os.path.join(config['output_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'config': config,
            'results': {k: {
                'best_params': v['best_params'],
                'metrics': v['metrics'],
                'best_val_accuracy': v['best_val_accuracy']
            } for k, v in results.items()},
            'best_classifier': best_classifier[0],
            'feature_extraction_time': {
                'train': train_time,
                'val': val_time,
                'test': test_time
            }
        }, f, indent=2)
    
    print(f"\nTraining summary saved to {summary_path}")
    print("\n🎉 Training complete!")

if __name__ == '__main__':
    main()
