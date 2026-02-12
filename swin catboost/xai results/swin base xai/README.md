# 🔍 SHAP Explainable AI - Swin Base + MobileViT Hybrid

This folder contains the SHAP (SHapley Additive exPlanations) analysis for the **Swin-Base + MobileViT Hybrid** model optimized with Optuna TPE.

## 📊 SHAP Metrics & Visualizations

### 1. **Mean Absolute SHAP** - Global Feature Importance
**Purpose**: Quantify the average contribution of each feature to model predictions across the entire dataset.

**Key Insight**: 
- Identifies which features (from the 256-dimensional hybrid representation) have the most significant impact on predictions
- Measures importance regardless of direction (positive or negative influence)
- Helps understand which learned representations are most critical for classification

**How it works for images**:
- Each of the 256 features represents a learned pattern from the hybrid Swin-Base + MobileViT model
- SHAP assigns contribution values to each feature dimension
- Mean absolute value shows overall importance magnitude
- Higher values = more influential features across the dataset

**Mathematical Definition**:
```
Mean Absolute SHAP = (1/N) × Σ|SHAP_value_i|
```
where N = number of samples, SHAP_value_i = contribution of feature for sample i

---

### 2. **Summary Plot** - Distribution of Feature Importance
**Purpose**: Visual overview of how features contribute across all samples and classes.

**Key Insight**:
- Shows not just WHICH features are important, but HOW they behave
- Displays the spread and density of SHAP values for each feature
- Color-coded dots indicate feature values (low to high)
- Reveals patterns like:
  - Features that consistently push predictions in one direction
  - Features with variable impact across samples
  - Outliers and edge cases
  - Feature interactions and dependencies

**Interpretation**:
- **Y-axis**: Features sorted by importance (top = most important)
- **X-axis**: SHAP value (contribution magnitude and direction)
  - Positive values → increase prediction probability
  - Negative values → decrease prediction probability
- **Color**: Feature value (blue = low, pink/red = high)
- **Density**: How spread out the contributions are
  - Tight cluster → consistent behavior
  - Wide spread → context-dependent behavior

---

### 3. **Force Plot** - Local Explanation for Individual Predictions
**Purpose**: Explain exactly how a specific prediction was made.

**Key Insight**:
- Shows the journey from base value (average prediction) to actual prediction
- Each feature's contribution is visualized as an arrow or bar
- **Red features**: Push prediction UP (positive contribution)
- **Blue features**: Push prediction DOWN (negative contribution)
- **Bar size**: Magnitude of contribution

**Components**:
- **Base value**: Average model output across the dataset (starting point)
- **Output value**: The model's prediction for this specific sample (endpoint)
- **Feature contributions**: Individual SHAP values showing each feature's push/pull effect
- **Net effect**: Sum of all contributions moves from base to output

**Use Cases**:
- Debugging misclassifications
- Understanding high-confidence vs low-confidence predictions
- Validating that model uses medically relevant features
- Building trust with clinicians

---

## 🎯 Model Details

- **Architecture**: Swin-Base (ImageNet-1K) + MobileViT (Kvasir-pretrained)
- **Feature Extraction**:
  - Swin-Base: **1024-dimensional** features (larger than Swin-Small)
  - MobileViT: **384-dimensional** features
  - Combined: 1408-dim → 256-dim (via learned fusion layer)
- **Classifier**: CatBoost (Optuna-optimized hyperparameters)
- **Dataset**: Kvasir-v2 (8,000 images, 8 GI disease classes)
- **Training Split**: 70% train, 15% validation, 15% test

### Performance Comparison:
| Model | Test Accuracy | AUROC | Parameters |
|-------|---------------|-------|------------|
| Swin-Base Hybrid | ~92-93% | ~99.5% | Larger backbone |
| Swin-Small Hybrid | 92.17% | 99.49% | Smaller backbone |

*Swin-Base provides richer features but requires more computation*

---

## 📂 Files Required

To run the SHAP analysis notebook, you need:

1. **`SHAP_XAI_Swin_Base_Optuna.ipynb`** - Main Colab notebook
2. **`model.py`** - MobileViT architecture definition
3. **`hybrid_model.py`** - Hybrid model class (supports both Swin-Base and Swin-Small)
4. **`swin_base_kvasir_features.npz`** - Extracted features (from `swin base/optuna tpe swin base/`)
5. **`swin_base_kvasir_optuna.cbm`** - Trained CatBoost model (from `swin base/optuna tpe swin base/`)
6. **`mobilevit_kvasir_v2_best_optuna.pth`** - MobileViT weights (from root directory)

---

## 🚀 How to Run

### Step 1: Open in Google Colab
1. Upload `SHAP_XAI_Swin_Base_Optuna.ipynb` to Google Colab
2. Enable GPU: **Runtime → Change runtime type → GPU (T4 or better)**
   - ⚠️ GPU is highly recommended for faster SHAP computation

### Step 2: Install Dependencies
Run the first cell to install all required packages:
```bash
pip install shap==0.43.0 torch torchvision catboost numpy pandas matplotlib seaborn scikit-learn pillow tqdm
```

### Step 3: Upload Files
When prompted, upload the 5 required files listed above.
- **Tip**: You can select all files at once in the file picker

### Step 4: Download Dataset
The notebook will automatically download the Kvasir-v2 dataset (~200MB).
- This only happens once; subsequent runs will skip if already downloaded

### Step 5: Run Analysis
Execute all cells sequentially. The notebook will:
1. ✅ Load the trained model and features
2. ✅ Initialize SHAP TreeExplainer (optimized for CatBoost)
3. ✅ Compute SHAP values for the test set (~1200 samples)
4. ✅ Generate all visualizations (25+ plots)
5. ✅ Create a comprehensive summary report
6. ✅ Package results into a downloadable ZIP

**Estimated Runtime**: 5-10 minutes on GPU

### Step 6: Download Results
The final cell creates `SHAP_XAI_Swin_Base_Results.zip` with all visualizations.

---

## 📈 Expected Outputs

### Visualizations Generated:

1. **Mean Absolute SHAP** (1 plot):
   - `mean_absolute_shap_swin_base.png` - Bar chart of top 20 features with values

2. **Summary Plots** (9 plots):
   - `shap_summary_plot_all_classes_swin_base.png` - Combined view across all classes
   - 8 per-class summary plots:
     - `shap_summary_dyed_lifted_polyps_swin_base.png`
     - `shap_summary_dyed_resection_margins_swin_base.png`
     - `shap_summary_esophagitis_swin_base.png`
     - `shap_summary_normal_cecum_swin_base.png`
     - `shap_summary_normal_pylorus_swin_base.png`
     - `shap_summary_normal_z_line_swin_base.png`
     - `shap_summary_polyps_swin_base.png`
     - `shap_summary_ulcerative_colitis_swin_base.png`

3. **Force Plots** (10 plots):
   - Individual prediction explanations
   - Includes high and low confidence samples from each class
   - Shows correct and incorrect predictions

4. **Waterfall Plots** (5 plots):
   - Alternative view of individual predictions
   - Shows cumulative feature effects step-by-step

5. **Heatmap** (1 plot):
   - `feature_importance_heatmap_swin_base.png` - Class-specific feature importance matrix

**Total**: ~25 high-resolution (300 DPI) visualizations

---

## 🔬 Understanding the Results

### What Makes a Feature Important?

In the Swin-Base hybrid model, each of the 256 features represents a learned combination of:

1. **Swin-Base Contributions** (1024-dim → compressed):
   - **Global context**: Long-range dependencies via shifted window attention
   - **Hierarchical features**: Multi-scale representations (patch → window → global)
   - **Semantic understanding**: High-level disease patterns

2. **MobileViT Contributions** (384-dim → compressed):
   - **Local details**: Fine-grained texture and structure
   - **Efficient representations**: Lightweight but expressive features
   - **Domain-specific knowledge**: Pretrained on Kvasir GI images

3. **Fusion Layer** (1408-dim → 256-dim):
   - Learned combination of both backbones
   - Removes redundancy
   - Emphasizes complementary information

### Interpreting SHAP Values:

| SHAP Value | Meaning | Example |
|------------|---------|---------|
| **Large Positive** | Strongly increases class probability | Feature 42 = +0.85 → "This looks like a polyp" |
| **Large Negative** | Strongly decreases class probability | Feature 17 = -0.72 → "This is NOT normal tissue" |
| **Near Zero** | Minimal impact on this prediction | Feature 99 = +0.03 → Irrelevant for this sample |
| **Consistent across samples** | Reliable discriminative feature | Feature 5 always high for ulcerative colitis |
| **Variable across samples** | Context-dependent feature | Feature 23 important only for certain polyp types |

### Clinical Relevance:

SHAP helps answer critical questions:
- ✅ "Why did the model classify this polyp as malignant?"
- ✅ "Which visual features distinguish ulcerative colitis from normal tissue?"
- ✅ "Are the model's decisions based on medically relevant patterns?"
- ✅ "Can we trust this prediction for clinical use?"
- ✅ "What happens if we remove this feature?"

### Example Insights:

From the analysis, you might discover:
- **Feature 12** consistently activates for redness/inflammation (esophagitis, ulcerative colitis)
- **Feature 34** detects polyp boundaries and shapes
- **Feature 67** captures normal tissue texture patterns
- **Features 100-120** (from MobileViT) focus on fine-grained details
- **Features 1-50** (from Swin-Base) capture global context

---

## 📊 Additional Analysis

The notebook includes:

### 1. Class-Specific Feature Rankings
For each of the 8 classes, identifies the top 10 most discriminative features.

### 2. Feature Importance Concentration
Calculates coefficient of variation to determine if importance is:
- **Concentrated**: Few features dominate (high CV)
- **Distributed**: Many features contribute equally (low CV)

### 3. Generalization Analysis
Compares train/val/test accuracy to assess:
- Overfitting risk
- Model stability
- Feature reliability

### 4. Confidence Analysis
Examines SHAP patterns for:
- High-confidence correct predictions
- Low-confidence correct predictions
- Misclassifications

---

## 💡 Tips & Best Practices

### Running the Notebook:
1. **GPU is essential**: CPU runtime will be very slow for SHAP computation
2. **Run cells in order**: Don't skip cells; they build on each other
3. **Check file uploads**: Verify all 5 files uploaded successfully
4. **Monitor memory**: Close other Colab notebooks to avoid OOM errors

### Interpreting Results:
1. **Start with summary plots**: Get overall picture before diving into individual predictions
2. **Compare classes**: Look for features unique to specific diseases
3. **Validate with domain knowledge**: Do important features make medical sense?
4. **Check edge cases**: Examine force plots for misclassifications
5. **Look for patterns**: Consistent features across samples are more reliable

### Common Issues:
- **"Model file not found"**: Ensure `swin_base_kvasir_optuna.cbm` is uploaded
- **"SHAP values shape mismatch"**: Verify features file matches model
- **"Out of memory"**: Reduce batch size or use smaller test set
- **"Plots not showing"**: Run `shap.initjs()` before force plots

---

## 📚 References

### SHAP Framework:
- Lundberg & Lee (2017) - ["A Unified Approach to Interpreting Model Predictions"](https://arxiv.org/abs/1705.07874)
- Lundberg et al. (2020) - ["From local explanations to global understanding with explainable AI for trees"](https://www.nature.com/articles/s42256-019-0138-9)

### Model Architectures:
- **Swin Transformer**: Liu et al. (2021) - ["Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"](https://arxiv.org/abs/2103.14030)
- **MobileViT**: Mehta & Rastegari (2021) - ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"](https://arxiv.org/abs/2110.02178)
- **CatBoost**: Prokhorenkova et al. (2018) - ["CatBoost: unbiased boosting with categorical features"](https://arxiv.org/abs/1706.09516)

### Dataset:
- **Kvasir-v2**: Pogorelov et al. (2017) - ["KVASIR: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection"](https://dl.acm.org/doi/10.1145/3083187.3083212)

---

## 🎓 Understanding SHAP Theory

### Shapley Values (Game Theory):
SHAP is based on Shapley values from cooperative game theory:
- **Players**: Features
- **Game**: Prediction task
- **Payout**: Model output
- **Fair distribution**: Each feature gets credit proportional to its contribution

### TreeExplainer Algorithm:
For tree-based models like CatBoost:
- **Exact computation**: No approximation needed
- **Fast**: Polynomial time complexity
- **Consistent**: Satisfies local accuracy and missingness properties
- **Additive**: SHAP values sum to (prediction - base_value)

### Why SHAP for Medical AI?
- ✅ **Model-agnostic**: Works with any model type
- ✅ **Theoretically grounded**: Based on solid mathematical foundation
- ✅ **Locally accurate**: Explains individual predictions precisely
- ✅ **Globally consistent**: Aggregates to feature importance
- ✅ **Clinically interpretable**: Visual explanations doctors can understand

---

## 📧 Support & Troubleshooting

### Common Questions:

**Q: Why are SHAP values different from feature importance?**
A: Feature importance is global (averaged across all samples). SHAP provides both local (per-sample) and global (aggregated) explanations.

**Q: Can I use this for other datasets?**
A: Yes, but you'll need to retrain the model and extract features for your dataset.

**Q: How do I know if a feature is "important enough"?**
A: Compare to the mean absolute SHAP. Features >2x mean are highly important.

**Q: What if force plots are too cluttered?**
A: Use waterfall plots instead, or increase `max_display` parameter.

**Q: Can I visualize SHAP on the original images?**
A: Not directly, since we work with extracted features. For pixel-level explanations, use GradCAM or attention maps.

### Getting Help:
1. Check that all dependencies installed correctly
2. Verify GPU is enabled and detected
3. Ensure model and features are from the same training run
4. Review the summary report for any warnings
5. Compare your results with the expected outputs above

---

## 🔄 Comparison: Swin-Base vs Swin-Small

| Aspect | Swin-Base | Swin-Small |
|--------|-----------|------------|
| **Feature Dimension** | 1024 | 768 |
| **Parameters** | ~88M | ~50M |
| **Computation** | Higher | Lower |
| **Accuracy** | ~92-93% | 92.17% |
| **Feature Richness** | More expressive | More efficient |
| **SHAP Insights** | Finer-grained | More generalizable |

**Recommendation**: Use Swin-Base for maximum accuracy, Swin-Small for efficiency.

---

**Status**: ✅ Ready to run in Google Colab  
**Last Updated**: December 2025  
**Model Version**: Optuna TPE Optimized  
**SHAP Version**: 0.43.0
