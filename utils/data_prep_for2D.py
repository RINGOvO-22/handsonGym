import pandas as pd
import numpy as np
from sklearn import preprocessing

def load_data(
    file_loc: str,
    feature_cols: list[str] | None = None,
    label_col: str = 'label',
    test_size: float = 0.2,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 CSV 里读数据，标准化、加 bias 列、标签 -1→0，再随机拆分 train/test。

    Parameters
    ----------
    file_loc : str
        CSV 路径，第一行必须有 [feature1, feature2, ..., label]。
    feature_cols : list[str] | None
        特征列名列表；若为 None，则自动取除 label 以外的所有列。
    label_col : str
        标签列名，默认 'label'。
    test_size : float
        测试集比例，0~1。
    seed : int | None
        随机种子，若 None 则不固定。

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    # 1) 读 CSV，不用把任何列当索引
    data = pd.read_csv(file_loc)

    # 2) 选 label，并把 -1 → 0
    if label_col not in data.columns:
        raise KeyError(f"No such label column '{label_col}'，columns exist: {data.columns.tolist()}")
    y = data[label_col].replace(-1, 0).to_numpy()

    # 3) 选特征列
    if feature_cols is None:
        feature_cols = [c for c in data.columns if c != label_col]
    X = data[feature_cols].to_numpy()

    # 4) 标准化
    X = preprocessing.scale(X)

    # 5) 加 bias 列
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    # 6) 随机打乱
    if seed is not None:
        np.random.seed(seed)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # 7) 拆分
    m = len(X)
    split = int(m * (1 - test_size))
    return X[:split], y[:split], X[split:], y[split:]

# Example usage
train_x, train_y, test_x, test_y = load_data(
    'data/generated_2D_data.csv',
    feature_cols=['feature1','feature2'],
    label_col='label',
    test_size=0.3,
    seed=42
)

print("train_x shape:", train_x.shape)
print("train_y unique:", np.unique(train_y))
print("test_x shape:", test_x.shape)
print("test_y unique:", np.unique(test_y))
# 统计标签分布
from collections import Counter
print("Train:", Counter(train_y))
print("Test: ", Counter(test_y))
