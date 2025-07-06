import pandas as pd
import numpy as np


def fill_na_with_default(df):
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].fillna(0).astype(dtype)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].fillna(0.0).astype(dtype)
        elif pd.api.types.is_bool_dtype(dtype):
            df[col] = df[col].fillna(False).astype(dtype)
        elif pd.api.types.is_object_dtype(dtype):
            df[col] = df[col].fillna('').astype(str)
    return df


def get_min_max_from_train(df, exclude_columns=[]):
    """
    获取训练集中所有数值列的 min/max 值用于后续归一化
    
    参数:
        df: 训练集 DataFrame
        exclude_columns: 不参与归一化的列名列表（如标签列）
    
    返回:
        min_vals: 各列最小值
        max_vals: 各列最大值
    """
    min_vals = {}
    max_vals = {}
    for col in df.columns:
        if col in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            min_vals[col] = df[col].min()
            max_vals[col] = df[col].max()
    return min_vals, max_vals


def normalize_with_min_max(df, min_vals, max_vals, exclude_columns=[]):
    """
    使用指定的 min/max 对 DataFrame 归一化
    
    参数:
        df: 待归一化的 DataFrame
        min_vals: 每列的最小值字典
        max_vals: 每列的最大值字典
        exclude_columns: 不参与归一化的列名列表
    
    返回:
        归一化后的 DataFrame
    """
    df_normalized = df.copy()
    for col in df_normalized.columns:
        if col in exclude_columns or col not in min_vals:
            continue
        if pd.api.types.is_numeric_dtype(df_normalized[col]):
            if max_vals[col] > min_vals[col]:
                df_normalized[col] = (df_normalized[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
                # 截断到 [0, 1] 范围
                df_normalized[col] = df_normalized[col].clip(0.0, 1.0)
            else:
                df_normalized[col] = 0.0  # 所有值相同，归为 0
    return df_normalized


def data_process():
    # 数据路径
    datapath = 'data/GiveMeSomeCredit/'

    # 加载并预处理训练集
    trainData = pd.read_csv(datapath + 'cs-training.csv')
    trainData = trainData.drop(columns=['Unnamed: 0'])  # 删除第一列
    trainData = fill_na_with_default(trainData)

    # 获取训练集的 min/max
    min_vals, max_vals = get_min_max_from_train(trainData, exclude_columns=['SeriousDlqin2yrs'])

    # 对训练集归一化
    # trainData = normalize_with_min_max(trainData, min_vals, max_vals, exclude_columns=['SeriousDlqin2yrs'])

    # 保存归一化后的训练集
    trainData.to_csv("data/ProcessedData/cs-training-processed.csv", index=False)
    print("Train data normalized and saved.")
    print("Min:\n", trainData.min())
    print("Max:\n", trainData.max())

    # 加载并预处理测试集
    testData = pd.read_csv(datapath + 'cs-test.csv')
    testData = testData.drop(columns=['Unnamed: 0'])  # 删除第一列
    testData = fill_na_with_default(testData)

    # 使用训练集的 min/max 对测试集归一化
    # testData = normalize_with_min_max(testData, min_vals, max_vals, exclude_columns=['SeriousDlqin2yrs'])

    # 保存归一化后的测试集
    testData.to_csv("data/ProcessedData/cs-test-processed.csv", index=False)
    print("Test data normalized with train's min/max and saved.")
    print("Min:\n", testData.min())
    print("Max:\n", testData.max())

def data_process_v2():
    """
    处理数据集，填充缺失值并归一化
    """
    datapath = 'data/GiveMeSomeCredit/'

    # 加载训练集
    trainData = pd.read_csv(datapath + 'cs-training.csv')
    trainData = trainData.drop(columns=['Unnamed: 0'])  # 删除第一列
    trainData = fill_na_with_default(trainData)
    # 获取训练集的 min/max
    min_vals, max_vals = get_min_max_from_train(trainData, exclude_columns=['SeriousDlqin2yrs'])
    # 对训练集归一化
    trainData = normalize_with_min_max(trainData, min_vals, max_vals, exclude_columns=['SeriousDlqin2yrs'])
    # 保存归一化后的训练集
    trainData.to_csv("data/ProcessedData/cs-training-processed.csv", index=False)
    print("\nTrain data normalized and saved.")
    print("Min:\n", trainData.min())
    print("Max:\n", trainData.max())

    # 加载测试集
    testData = pd.read_csv(datapath + 'cs-test.csv')
    testData = testData.drop(columns=['Unnamed: 0'])  # 删除第一列
    testData = fill_na_with_default(testData)
    # 使用训练集的 min/max 对测试集归一化
    testData = normalize_with_min_max(testData, min_vals, max_vals, exclude_columns=['SeriousDlqin2yrs'])
    # 保存归一化后的测试集
    testData.to_csv("data/ProcessedData/cs-test-processed.csv", index=False)
    print("\nTest data normalized with train's min/max and saved.")
    print("Min:\n", testData.min())
    print("Max:\n", testData.max())

if __name__ == "__main__":
    # data_process() # unnormalized
    data_process_v2() # normalized
    print("Data processing completed.")