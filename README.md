# Grassmann-manifold-for-voiceprint-data
Grassmann manifold for voiceprint data

 import numpy as np
from scipy.linalg import svd
import scipy.io
import numpy as np
from scipy.linalg import eig
from scipy.linalg import svd


######读者根据实际需要更改输入；
###### 为了输入方便，此代码没有使用ten-fold cross 验证；以独立GIS采样分组，SMOTE用于增强训练数据，在不同随机种子下；参数敏感性实验中，选择不同的随机种子。
###### 这里只提供 project kernel，BCkernel 和canonical kernel读者可以自行更改，最后再加入经验核对齐操作即可。

更新特异性指标，
def calculate_specificity(y_true, y_pred, labels=None):
    """
    计算多分类任务的特异性。

    Parameters
    ----------
    y_true : array-like
        真实标签。
    y_pred : array-like
        预测标签。
    labels : array-like, optional
        类别顺序，例如 np.arange(20)。

    Returns
    -------
    specificity_per_class : np.ndarray
        每个类别的特异性。
    macro_specificity : float
        各类别特异性的算术平均值。
    micro_specificity : float
        汇总所有类别的 TN 和 FP 后计算的特异性。
    """
    mcm = multilabel_confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # 每个类别对应：
    # [[TN, FP],
    #  [FN, TP]]
    tn = mcm[:, 0, 0].astype(float)
    fp = mcm[:, 0, 1].astype(float)

    specificity_per_class = np.divide(
        tn,
        tn + fp,
        out=np.full_like(tn, np.nan, dtype=float),
        where=(tn + fp) != 0
    )

    macro_specificity = np.nanmean(specificity_per_class)

    micro_specificity = (
        tn.sum() / (tn.sum() + fp.sum())
        if (tn.sum() + fp.sum()) > 0
        else np.nan
    )

    return specificity_per_class, macro_specificity, micro_specificity
