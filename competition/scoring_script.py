"""
scoring_script.py

Comprehensive evaluation script for GNN predictions.
Computes 3 challenging metrics: Macro F1, MCC, and Balanced Accuracy.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def score_predictions(y_true, y_pred, y_proba=None):
    """
    Evaluate predictions with multiple metrics.
    
    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Hard predictions (0 or 1)
        y_proba: Soft predictions/probabilities (optional, for ROC-AUC, AP)
    
    Returns:
        Dictionary of metrics
    """
    
    metrics = {}
    
    # ===== BASIC METRICS =====
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    
    # ===== DIFFICULT METRIC #1: MACRO F1-SCORE =====
    # Equal weight per class (penalizes minority class errors)
    # metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    # metrics['f1_binary'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred) 
   
    return metrics

def print_metrics(metrics, name="Evaluation Results"):
    """Pretty-print metrics."""
    print("\n" + "="*60)
    print(f"  {name}")
    print("="*60)
    
    print(f"\n📊 Basic Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  F1-score:           {metrics['f1']:.4f}")
    

def evaluate_submission(submission_path, ground_truth_path=None):
    """
    Evaluate a submission CSV against ground truth.
    
    Args:
        submission_path: Path to predictions CSV
        ground_truth_path: Path to ground truth CSV (if available)
    """
    # Load submission
    submission = pd.read_csv(submission_path)

    # Normalize expected columns
    if "id" not in submission.columns or "y_pred" not in submission.columns:
        print("❌ Submission must have ['id','y_pred']  columns")
        return None
    
    # Check if ground truth available
    if ground_truth_path is None:
        print("⚠️  No ground truth provided. Cannot evaluate.")
        print(f"   Submission has {len(submission)} predictions")
        print(f"   Classes: {submission['target'].unique()}")
        return None
    
    # Load ground truth
    ground_truth = pd.read_csv(ground_truth_path)
    
    # Merge on node_id
    # merged = pd.merge(ground_truth, submission, on='node_id', suffixes=('_true', '_pred'))
    merged = pd.merge(ground_truth, submission, on='id', suffixes=('_true', '_pred'))
    print(" number of matching ids between ground_truth and submission" , len(merged))
    if len(merged) == 0:
        print("❌ No matching node_ids between submission and ground truth")
        return None
    
    # y_true = merged['target_true'].values
    # y_pred_raw = merged['target_pred'].values

    y_true = merged['y_true'].values
    y_pred_raw = merged['y_pred'].values

    # If probabilities, threshold at 0.5
    if y_pred_raw.dtype.kind in {"f", "c"}:
        y_pred = (y_pred_raw >= 0.5).astype(int)
    else:
        y_pred = y_pred_raw.astype(int)
    
    # Evaluate
    metrics = score_predictions(y_true, y_pred)
    print_metrics(metrics, name="Submission Evaluation")
    
    return metrics

if __name__ == "__main__":
    import sys
    
    print("\n🎯 GNN Challenge Scoring Script")
    print("="*60)
    
    # Example usage
    submission_file = "submissions/inbox/example_team/example_run/predictions.csv"
    ground_truth_file = "data/test_labels.csv"  # True labels for test set created in the workflow by decoding secret csv
    
    if len(sys.argv) > 1:
        submission_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        ground_truth_file = sys.argv[2]
    
    if os.path.exists(submission_file):
        print(f"\n📂 Submission: {submission_file}")
        print(f"   Ground truth: {ground_truth_file}")
        metrics = evaluate_submission(submission_file, ground_truth_file)
    else:
        print(f"❌ Submission file not found: {submission_file}")
        print(f"\nUsage: python scoring_script.py <submission_file.csv> [ground_truth_file.csv]")
        print(f"\nExample:")
        print(f"  python scoring_script.py submissions/inbox/my_team/run_001/predictions.csv")
