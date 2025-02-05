import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rs
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import Point
from datetime import datetime
from sklearn.utils import resample
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn import metrics
import json

def pretty_print_metrics(val_df_file):
    val_df = pd.read_csv(val_df_file)
    y_pred = val_df.y_pred
    y_true = val_df.y_true
    class_names = list(set(y_true))
    f1_scores = metrics.f1_score(y_true, y_pred, labels=class_names, average=None)
    avg_f1_score = metrics.f1_score(y_true, y_pred, average="weighted")
    recall = metrics.recall_score(y_true, y_pred, labels=class_names, average=None)
    precision = metrics.precision_score(y_true, y_pred, labels=class_names, average=None)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    
    print("Classification accuracy {:.1f}%".format(100 * accuracy))
    print("Classification F1-score {:.1f}%".format(100 * avg_f1_score))
    print()
    print("             Class              =  F1  | Recall | Precision")
    print("         --------------------------------------------------")
    
    results = []
    for idx, lulctype in enumerate(class_names):
        f1, rec, prec = f1_scores[idx] * 100, recall[idx] * 100, precision[idx] * 100
        results.append([lulctype, f1, rec, prec])
        print("         * {0:20s} = {1:2.1f} |  {2:2.1f}  | {3:2.1f}".format(lulctype, f1, rec, prec))
    
    results.append(["Overall", avg_f1_score * 100, "-", "-", accuracy * 100])
    df_results = pd.DataFrame(results, columns=['Class', 'F1 Score (%)', 'Recall (%)', 'Precision (%)', 'Accuracy (%)'])
    df_results['Run'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') 
    outf="../../data/validation/accuracy_assessment_log.csv"
    df_results.to_csv(outf, 
                      mode='a', 
                      index=False, 
                      header=not pd.io.common.file_exists(outf))
    

def ci_error_adjustments(val_df_file,
                      outfile,
                      n_iterations=1000, 
                      confidence_level=95,
                      ):
    """
    Perform bootstrapping to compute precision/recall confidence intervals for 
    class-wise accuracy.Calculates mean and margin of error in order to present 
    them as x% ± y%.
    
    The 2.5% and 97.5% percentiles (for a 95% confidence level) are computed 
    from the bootstrap samples for each class.

    An error adjustment "adj" is calculated for each class according to the following logic:
    If recall < precision, adj < 1, reducing the mapped area (overrepresentation correction).
    If recall > precision, adj > 1, increasing the mapped area (underrepresentation correction).
    If recall ≈ precision, adj ≈ 1, meaning minimal adjustment.
    
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        class_labels (list): List of class labels to compute metrics for. (this will be str)
        n_iterations (int): Number of bootstrap resamples.
        confidence_level (int): Confidence level percentage (e.g., 95 for 95% CI).
        
    Returns:
        dict: A dictionary containing mean and margin of error for recall and precision.
        
    """
    val_df = pd.read_csv(val_df_file)
    y_pred = val_df.y_pred
    y_true = val_df.y_true
    class_names = list(set(y_true))

    boot = {label: {"recall": [], "precision": [], "adj": []} for label in class_names}
    boot["overall_accuracy"] = []

    for _ in range(n_iterations):
        
        # Resample with replacement
        y_true_sample, y_pred_sample = resample(y_true, y_pred)
        
        # Calculate overall accuracy
        overall_acc = metrics.accuracy_score(y_true_sample, y_pred_sample)
        boot["overall_accuracy"].append(overall_acc)
        
        for label in class_names:
            # Compute recall and precision for the current class
            recall = metrics.recall_score(y_true_sample, y_pred_sample, labels=[label], average=None)[0]
            precision = metrics.precision_score(y_true_sample, y_pred_sample, labels=[label], average=None)[0]
    
            boot[label]["recall"].append(recall)
            boot[label]["precision"].append(precision)

    # Compute percentiles for the confidence intervals
    # For a 95% CI, alpha gives 2.5%
    summary = {}
    alpha = (100 - confidence_level) / 2  

    accuracy_lower = np.percentile(boot["overall_accuracy"], alpha)
    accuracy_upper = np.percentile(boot["overall_accuracy"], 100 - alpha)
    accuracy_mean = (accuracy_lower + accuracy_upper) / 2
    accuracy_margin = (accuracy_upper - accuracy_lower) / 2
    summary["overall_accuracy"] = (accuracy_mean, accuracy_margin)

    for label in class_names:
        
        # Compute recall confidence interval then calc mean and margin of error
        recall_lower = np.percentile(boot[label]["recall"], alpha)
        recall_upper = np.percentile(boot[label]["recall"], 100 - alpha)
        recall_mean = (recall_lower + recall_upper) / 2
        recall_margin = (recall_upper - recall_lower) / 2

        # Compute precision confidence interval then calc mean and margin of error
        precision_lower = np.percentile(boot[label]["precision"], alpha)
        precision_upper = np.percentile(boot[label]["precision"], 100 - alpha)
        precision_mean = (precision_lower + precision_upper) / 2
        precision_margin = (precision_upper - precision_lower) / 2

        # Compute area assessment adjustment
        adj = recall_mean / precision_mean if precision_mean > 0 else 1  # Avoid division by zero

        summary[label] = {
            "recall": (recall_mean, recall_margin),
            "precision": (precision_mean, precision_margin),
            "adj": adj
        }
    print(f"Saving error metrics to {outfile}")
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nBootstrap Confidence Interval Summary (Mean ± Margin of Error):")
    for n in class_names:
        recall_mean, recall_margin = summary[n]["recall"]
        precision_mean, precision_margin = summary[n]["precision"]
        
        print(f"Class {n}:")
        print(f"  Recall    = {recall_mean:.2%} ± {recall_margin:.2%}")
        print(f"  Precision = {precision_mean:.2%} ± {precision_margin:.2%}")

    # Print overall accuracy with margin of error
    overall_mean, overall_margin = summary["overall_accuracy"]
    print(f"\nOverall Accuracy: {overall_mean:.2%} ± {overall_margin:.2%}")

    return summary