import pandas as pd
import numpy as np

# hyperparameters
test_label_threshold = 0.5

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

    # trainData & testData has the same column struction, where the 1st column is the target label
    # N.B., the 1st column of the testData is just a placeholder.
    # N.B., the real target label is saved in sampleEvtry.csv
    return trainData, testData

def data_process_v2():
    """
    处理数据集，填充缺失值并归一化
    """
    datapath = 'data/GiveMeSomeCredit/'

    # 加载训练集
    trainData = pd.read_csv(datapath + 'cs-training.csv')
    trainData = trainData.drop(columns=['Unnamed: 0'])  # 删除第一列
    trainData = fill_na_with_default(trainData)

    # 观察最值与均值
    # 获取训练集的 min/max
    min_vals, max_vals = get_min_max_from_train(trainData, exclude_columns=['SeriousDlqin2yrs'])
    train_means = trainData.mean(numeric_only=True)

    # 打印清晰格式的统计信息
    print("\n=== Pre-normalization statistics (train data) ===")
    stat_df = pd.DataFrame({
        'Min': pd.Series(min_vals),
        'Max': pd.Series(max_vals),
        'Mean': train_means
    })
    print(stat_df.to_string(float_format='%.4f'))

    # 获取训练集的 min/max
    min_vals, max_vals = get_min_max_from_train(trainData, exclude_columns=['SeriousDlqin2yrs'])
    # 对训练集归一化
    trainData = normalize_with_min_max(trainData, min_vals, max_vals, exclude_columns=['SeriousDlqin2yrs'])
    # 保存归一化后的训练集
    trainData.to_csv("data/ProcessedData/cs-training-processed.csv", index=False)
    print("\nTrain data normalized and saved.")
    print("Min:\n", trainData.min())
    print("Max:\n", trainData.max())
    print("\n===================================================================\n")

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

    # trainData & testData has the same column struction, where the 1st column is the target label
    # N.B., the 1st column of the testData is just a placeholder.
    # N.B., the real target label is saved in sampleEvtry.csv
    return trainData, testData

def data_process_v3():
    """
    处理数据集，填充缺失值 + 剪裁极端值 + min-max归一化（推荐方式）
    """
    datapath = 'data/GiveMeSomeCredit/'

    # 加载训练集
    trainData = pd.read_csv(datapath + 'cs-training.csv')
    trainData = trainData.drop(columns=['Unnamed: 0'])  # 删除第一列
    trainData = fill_na_with_default(trainData)

    # 打印未归一化前统计
    min_vals, max_vals = get_min_max_from_train(trainData, exclude_columns=['SeriousDlqin2yrs'])
    train_means = trainData.mean(numeric_only=True)

    print("\n=== Pre-normalization statistics (train data) ===")
    stat_df = pd.DataFrame({
        'Min': pd.Series(min_vals),
        'Max': pd.Series(max_vals),
        'Mean': train_means
    })
    print(stat_df.to_string(float_format='%.4f'))

    # 分位数剪裁（对数值型列）
    for col in trainData.columns:
        if col == 'SeriousDlqin2yrs':
            continue
        if pd.api.types.is_numeric_dtype(trainData[col]):
            q_low, q_high = trainData[col].quantile([0.01, 0.99])
            trainData[col] = trainData[col].clip(q_low, q_high)

    # 再次获取剪裁后的 min/max（用于归一化）
    min_vals, max_vals = get_min_max_from_train(trainData, exclude_columns=['SeriousDlqin2yrs'])

    # 归一化
    trainData = normalize_with_min_max(trainData, min_vals, max_vals, exclude_columns=['SeriousDlqin2yrs'])

    # 保存训练集
    trainData.to_csv("data/ProcessedData/cs-training-processed.csv", index=False)
    print("\nTrain data clipped, normalized and saved.")
    print("Min:\n", trainData.min())
    print("Max:\n", trainData.max())

    print("\n===================================================================\n")

    # 加载测试集
    testData = pd.read_csv(datapath + 'cs-test.csv')
    testData = testData.drop(columns=['Unnamed: 0'])
    testData = fill_na_with_default(testData)

    # 对测试集进行与训练集一致的剪裁
    for col in testData.columns:
        if col == 'SeriousDlqin2yrs':
            continue
        if pd.api.types.is_numeric_dtype(testData[col]) and col in min_vals:
            q_low, q_high = testData[col].quantile([0.01, 0.99])
            testData[col] = testData[col].clip(q_low, q_high)

    # 用训练集的 min/max 对测试集归一化
    testData = normalize_with_min_max(testData, min_vals, max_vals, exclude_columns=['SeriousDlqin2yrs'])

    # 保存测试集
    testData.to_csv("data/ProcessedData/cs-test-processed.csv", index=False)
    print("\nTest data clipped, normalized and saved.")
    print("Min:\n", testData.min())
    print("Max:\n", testData.max())

    return trainData, testData

def label_distribution(trainData):
    print("\n================ label distribution check ==================")
    datapath = 'data/GiveMeSomeCredit/'

    # 加载 test label
    test_prob = pd.read_csv(datapath + 'sampleEntry.csv')
    # test_prob = test_prob.drop(columns=['Id'])  # 删除第一列
    print(f"\nNumber of test samples: {test_prob.shape[0]}")
    test_prob_subset = test_prob.loc[:]
    test_y_1_indices = test_prob.query(f'Probability >= {test_label_threshold}').index.to_list()
    print(f"Test_y = 1: {len(test_y_1_indices)}")
    # print(f"Indices of them: {test_y_1_indices}")

    train_subset = trainData.loc[:]
    train_y_1_indices = train_subset.query(f"SeriousDlqin2yrs == 1").index.to_list()
    print(f"\nNumber of training samples: {train_subset.shape[0]}")
    print(f"Train_y = 1: {len(train_y_1_indices)}\n")
    # print(f"Indices of them: {train_y_1_indices}")

    # train label statistics
    return

if __name__ == "__main__":
    trainData, testData = data_process_v3() # clip + min-max normalilzation
    label_distribution(trainData)
    print("Data processing completed.")