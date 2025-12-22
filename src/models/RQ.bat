#!/bin/bash

# ==============================================================================
# Script to Reproduce Experiments for "GMove" Paper
# ==============================================================================

# ----------------- CONFIGURATION -----------------
# PLEASE UPDATE THESE PATHS TO MATCH YOUR ACTUAL FILE LOCATIONS
SYNTHETIC_DATA="./data/train_data.csv"   # Path to the large synthetic dataset
REAL_DATA_DIR="./data/real_projects/"    # Directory containing ant.csv, weka.csv, etc.
# -------------------------------------------------

echo "======================================================================"
echo "Starting Reproduction of Experimental Results"
echo "======================================================================"

# Check if data exist
if [ ! -f "$SYNTHETIC_DATA" ]; then
    echo "Error: Synthetic dataset not found at $SYNTHETIC_DATA"
    echo "Please configure the SYNTHETIC_DATA path in the script."
    exit 1
fi

# ------------------------------------------------------------------------------
# EXPERIMENT 1: RQ1 & RQ4 - Baseline Models Comparison
# Tables V and VII in the paper
# ------------------------------------------------------------------------------
echo ""
echo ">>> Running Machine Learning Baselines (RQ1 & RQ4)..."
echo "Comparing: XGB, SVC, DT, RF, ExtraTrees, NB, LR"

echo "[1/7] Running XGBoost..."
python xgb.py "$SYNTHETIC_DATA"

echo "[2/7] Running SVC (Support Vector Machine)..."
python SVC.py "$SYNTHETIC_DATA"

echo "[3/7] Running Decision Tree..."
python DecisionTree.py "$SYNTHETIC_DATA"

echo "[4/7] Running Random Forest..."
python RandomForest.py "$SYNTHETIC_DATA"

echo "[5/7] Running Extra Trees..."
python ExtraTrees.py "$SYNTHETIC_DATA"

echo "[6/7] Running Naive Bayes..."
python NB.py "$SYNTHETIC_DATA"

echo "[7/7] Running Logistic Regression..."
python LogisticRegression.py "$SYNTHETIC_DATA"

echo ">>> Baseline experiments completed."

# ------------------------------------------------------------------------------
# EXPERIMENT 2: RQ2 - GMove Model Training and Validation
# Fig. 7 in the paper (Performance on Synthetic Dataset)
# ------------------------------------------------------------------------------
echo ""
echo ">>> Running GMove Model Training (RQ2)..."
echo "This will train the Deep Learning model on the synthetic dataset."

# main.py trains the model and outputs Accuracy, Precision, Recall, F1
python main.py "$SYNTHETIC_DATA"

echo ">>> GMove training completed."

# ------------------------------------------------------------------------------
# EXPERIMENT 3: RQ5 - Real-World Generalization
# Table VIII in the paper (Comparison on Qualitas.class benchmark)
# ------------------------------------------------------------------------------
echo ""
echo ">>> Running Real-World Project Evaluation (RQ5)..."
echo "Evaluating GMove on projects: Weka, Ant, FreeCol, JMeter, etc."

if [ ! -d "$REAL_DATA_DIR" ]; then
    echo "Warning: Real data directory not found at $REAL_DATA_DIR. Skipping RQ5."
else
    # test.py iterates through the list of projects defined in the code
    # and evaluates the pre-trained model on them.
    # Ensure 'config.save_path' in model.py points to the model saved in the previous step.
    python test.py "$REAL_DATA_DIR"
fi

echo ""
echo "======================================================================"
echo "All experiments finished."
echo "======================================================================"