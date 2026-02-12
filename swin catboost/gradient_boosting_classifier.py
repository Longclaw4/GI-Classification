import numpy as np
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pickle
import json

class GradientBoostingClassifier:
    """
    Wrapper for gradient boosting classifiers (XGBoost, LightGBM, CatBoost)
    with Optuna optimization support.
    """
    def __init__(self, classifier_type='xgboost', num_classes=8):
        self.classifier_type = classifier_type
        self.num_classes = num_classes
        self.model = None
        self.best_params = None
    
    def create_objective(self, X_train, y_train, X_val, y_val, sampler='tpe'):
        """
        Create Optuna objective function for hyperparameter optimization.
        sampler: 'tpe' for TPE or 'bohb' for BOHB
        """
        def objective(trial):
            if self.classifier_type == 'xgboost':
                params = {
                    'objective': 'multi:softprob',
                    'num_class': self.num_classes,
                    'eval_metric': 'mlogloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'tree_method': 'gpu_hist' if trial.suggest_categorical('use_gpu', [True]) else 'hist',
                    'predictor': 'gpu_predictor' if trial.suggest_categorical('use_gpu', [True]) else 'cpu_predictor'
                }
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
            elif self.classifier_type == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': self.num_classes,
                    'metric': 'multi_logloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'device': 'gpu' if trial.suggest_categorical('use_gpu', [True]) else 'cpu',
                    'verbose': -1
                }
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
                
            elif self.classifier_type == 'catboost':
                params = {
                    'loss_function': 'MultiClass',
                    'classes_count': self.num_classes,
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_seed': 42,
                    'task_type': 'GPU' if trial.suggest_categorical('use_gpu', [True]) else 'CPU',
                    'verbose': False
                }
                
                model = cb.CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            
            else:
                raise ValueError(f"Invalid classifier_type: {self.classifier_type}")
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
        
        return objective
    
    def optimize(self, X_train, y_train, X_val, y_val, n_trials=50, sampler='tpe'):
        """
        Optimize hyperparameters using Optuna.
        sampler: 'tpe' for TPE or 'bohb' for BOHB
        """
        print(f"\n{'='*60}")
        print(f"Optimizing {self.classifier_type.upper()} with {sampler.upper()} sampler")
        print(f"{'='*60}\n")
        
        # Choose sampler
        if sampler == 'tpe':
            sampler_obj = optuna.samplers.TPESampler(seed=42)
        elif sampler == 'bohb':
            sampler_obj = optuna.integration.BoTorchSampler(seed=42)
        else:
            raise ValueError(f"Invalid sampler: {sampler}. Choose 'tpe' or 'bohb'.")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler_obj,
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Optimize
        objective = self.create_objective(X_train, y_train, X_val, y_val, sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Store best parameters
        self.best_params = study.best_params
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best validation accuracy: {study.best_value:.4f}")
        print(f"Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return study
    
    def train_with_best_params(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train model with best parameters found by Optuna.
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimize() first.")
        
        print(f"\nTraining {self.classifier_type.upper()} with best parameters...")
        
        if self.classifier_type == 'xgboost':
            params = {
                'objective': 'multi:softprob',
                'num_class': self.num_classes,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                **{k: v for k, v in self.best_params.items() if k != 'use_gpu'}
            }
            if self.best_params.get('use_gpu', False):
                params['tree_method'] = 'gpu_hist'
                params['predictor'] = 'gpu_predictor'
            
            self.model = xgb.XGBClassifier(**params)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
            
        elif self.classifier_type == 'lightgbm':
            params = {
                'objective': 'multiclass',
                'num_class': self.num_classes,
                'metric': 'multi_logloss',
                'random_state': 42,
                'verbose': -1,
                **{k: v for k, v in self.best_params.items() if k != 'use_gpu'}
            }
            if self.best_params.get('use_gpu', False):
                params['device'] = 'gpu'
            
            self.model = lgb.LGBMClassifier(**params)
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set)
            
        elif self.classifier_type == 'catboost':
            params = {
                'loss_function': 'MultiClass',
                'classes_count': self.num_classes,
                'random_seed': 42,
                'verbose': False,
                **{k: v for k, v in self.best_params.items() if k != 'use_gpu'}
            }
            if self.best_params.get('use_gpu', False):
                params['task_type'] = 'GPU'
            
            self.model = cb.CatBoostClassifier(**params)
            eval_set = (X_val, y_val) if X_val is not None else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        
        print("Training complete!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.
        """
        if self.model is None:
            raise ValueError("Model not trained. Run train_with_best_params() first.")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Calculate AUROC
        y_test_bin = label_binarize(y_test, classes=range(self.num_classes))
        auroc = roc_auc_score(y_test_bin, y_pred_proba, average='macro', multi_class='ovr')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auroc': auroc
        }
        
        print(f"\n{'='*60}")
        print(f"{self.classifier_type.upper()} Test Results")
        print(f"{'='*60}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def save(self, filepath):
        """Save model and best parameters."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        # Save model
        if self.classifier_type == 'xgboost':
            self.model.save_model(f"{filepath}.json")
        elif self.classifier_type == 'lightgbm':
            self.model.booster_.save_model(f"{filepath}.txt")
        elif self.classifier_type == 'catboost':
            self.model.save_model(f"{filepath}.cbm")
        
        # Save best parameters
        with open(f"{filepath}_params.json", 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model and best parameters."""
        # Load parameters
        with open(f"{filepath}_params.json", 'r') as f:
            self.best_params = json.load(f)
        
        # Load model
        if self.classifier_type == 'xgboost':
            self.model = xgb.XGBClassifier()
            self.model.load_model(f"{filepath}.json")
        elif self.classifier_type == 'lightgbm':
            self.model = lgb.Booster(model_file=f"{filepath}.txt")
        elif self.classifier_type == 'catboost':
            self.model = cb.CatBoostClassifier()
            self.model.load_model(f"{filepath}.cbm")
        
        print(f"Model loaded from {filepath}")
