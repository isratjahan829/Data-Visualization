# Laptop Price Predictor

Predict a laptop's price from its specification. The assignment's 20 steps are split into four notebooks, in the
same order as the assignment document. All notebooks are already executed, with every output saved.

| Part | Notebook | Assignment steps | Reads | Writes |
|---|---|---|---|---|
| 1 | `Part1_Data_Loading_and_Feature_Extraction.ipynb` | 1 Import Library · 2 Data Loading · 3 Feature Extraction · 4 Data Info | `Cleaned_Laptop_data.csv` | `laptop_part1_features.csv` |
| 2 | `Part2_Missing_Data_and_Duplicates.ipynb` | 5 Checking missing data · 6 Missing Data Treatment · 7 Duplicate Checking · 8 Duplicate Removing | `laptop_part1_features.csv` | `laptop_part2_clean.csv` |
| 3 | `Part3_Visualization_and_Feature_Engineering.ipynb` | 9 Data Visualization · 10 Correlation with target · 11 Features and target separation · 12 Feature Selection · 13 Feature Scaling · 14 PCA · 15 Encoding | `laptop_part2_clean.csv` | `laptop_part3_model_ready.csv` |
| 4 | `Part4_Model_Training_and_Cross_Validation.ipynb` | 16 Train Test Split · 17 Model Preparation · 18 Model Accuracy · 19 Cross_val function · 20 cross_val Accuracy | `laptop_part3_model_ready.csv` | — |

All CSV files are in `data/`, so any part can be run on its own.

## Running in Google Colab
1. Open a notebook in Colab (File → Upload notebook, or open it from GitHub).
2. *Runtime → Run all.* If the input CSV is not in the Colab file panel, the notebook asks you to upload it
   (take it from `data/`).
3. At the end of Parts 1–3, the output CSV downloads automatically. Upload it into the next part.

Running locally (Jupyter / VS Code): keep the `data/` folder next to the notebooks and run them in order.

## Results
| Model | Test R² | Cross-val mean R² (5 folds) |
|---|---|---|
| Gradient Boosting | 0.652 | **0.612** ± 0.087 |
| Random Forest | 0.455 | 0.603 ± 0.100 |
| Linear Regression | 0.631 | 0.598 ± 0.050 |
| PCA + Linear Regression | 0.500 | 0.470 ± 0.051 |
| Decision Tree | 0.462 | 0.450 ± 0.101 |

Gradient Boosting is the best model. Price is driven mostly by SSD size, graphics card memory and brand.
