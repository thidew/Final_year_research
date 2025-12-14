
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

def save_plot(fig, title, folder):
    """Save plot to file"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    filename = title.lower().replace(" ", "_").replace("/", "-") + ".png"
    fig.savefig(folder / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {filename}")

def plot_confusion_matrix_custom(y_true, y_pred, classes, title, folder):
    """Plot custom confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_plot(fig, f"confusion_matrix_{title}", folder)

def plot_multiclass_roc(model, X_test, y_test, classes, title, folder):
    """Plot ROC curves for multiclass"""
    try:
        y_score = model.predict_proba(X_test)
    except AttributeError:
        print(f"Model {title} does not support predict_proba")
        return

    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        if n_classes == 2:
             # Binary case handling
             fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, 1]) # Use probability of positive class
        else:
             fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
             
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {title}')
    plt.legend(loc="lower right")
    save_plot(fig, f"roc_curve_{title}", folder)

def plot_precision_recall_curves(model, X_test, y_test, classes, title, folder):
    """Plot Precision-Recall curves"""
    try:
        y_score = model.predict_proba(X_test)
    except AttributeError:
        return

    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        if n_classes == 2:
             precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, 1])
        else:
             precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
             
        plt.plot(recall, precision, lw=2, label=f'{classes[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {title}')
    plt.legend(loc="best")
    save_plot(fig, f"pr_curve_{title}", folder)

def plot_learning_curves(estimator, X, y, title, folder):
    """Plot learning curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use fewer n_jobs and points for speed
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=1, 
        train_sizes=np.linspace(0.1, 1.0, 4),
        scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.title(f"Learning Curve: {title}")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    save_plot(fig, f"learning_curve_{title}", folder)

def plot_calibration_curve(model, X_test, y_test, classes, title, folder):
    """Plot Calibration Curve (Reliability Diagram)"""
    try:
        y_prob = model.predict_proba(X_test)
    except AttributeError:
        return
        
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        prob_pos = y_prob[:, i] if n_classes > 2 else y_prob[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test_bin[:, i], prob_pos, n_bins=10)
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{classes[i]}")

    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title(f'Calibration Curve: {title}')
    plt.legend(loc="upper left")
    save_plot(fig, f"calibration_curve_{title}", folder)

def plot_cumulative_gain_lift(model, X_test, y_test, classes, title, folder):
    """Plot Cumulative Gain and Lift Charts"""
    try:
        # scikit-plot is great but might not be installed. Basic manual implementation needed.
        import scikitplot as skplt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        skplt.metrics.plot_cumulative_gain(y_test, model.predict_proba(X_test), ax=ax)
        save_plot(fig, f"cumulative_gain_{title}", folder)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        skplt.metrics.plot_lift_curve(y_test, model.predict_proba(X_test), ax=ax)
        save_plot(fig, f"lift_curve_{title}", folder)
        
    except ImportError:
        print("scikit-plot not installed, skipping Gain/Lift charts")

def plot_class_prediction_error(y_true, y_pred, classes, title, folder):
    """Plot Class Prediction Error (Stacked Bar)"""
    # Create confusion matrix-like structure
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize rows to sum to 100%
    with np.errstate(divide='ignore', invalid='ignore'):
         cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
         cm_norm = np.nan_to_num(cm_norm) # Handle cases where a class has 0 samples
    
    # Plot stacked bars
    bottom = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        plt.bar(classes, cm_norm[:, i], bottom=bottom, label=f'Pred {cls}')
        bottom += cm_norm[:, i]
        
    plt.title(f"Class Prediction Error: {title}")
    plt.xlabel("True Class")
    plt.ylabel("Fraction of Predictions")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot(fig, f"class_prediction_error_{title}", folder)

def plot_residuals(y_true, y_pred, title, folder):
    """Plot Residuals (For Regression, but adapted if needed)"""
    pass # Not applicable for Classification tasks requested
