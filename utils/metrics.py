# utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
import seaborn as sns

def calculate_detailed_metrics(y_true, y_pred, y_prob=None):
    """计算详细的分类指标"""
    metrics = {}
    
    # 基础指标
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                               recall_score, roc_auc_score)
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    
    # AUC-ROC
    if y_prob is not None:
        if len(np.unique(y_true)) > 2:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        else:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
    
    # 入侵检测特定指标
    cm = confusion_matrix(y_true, y_pred)
    
    # 对于二分类（正常 vs 攻击）
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['f1_score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        # 检测率（Detection Rate）
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 误报率（False Alarm Rate）
        metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_prob, class_names, save_path=None):
    """绘制ROC曲线"""
    from sklearn.preprocessing import label_binarize
    
    # 二值化标签
    if len(np.unique(y_true)) > 2:
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        n_classes = y_true_bin.shape[1]
    else:
        y_true_bin = label_binarize(y_true, classes=[0, 1])
        n_classes = 1
    
    # 计算每个类别的ROC曲线
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()