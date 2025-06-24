import pandas as pd

def fill_na_with_default(df):
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            df[col] = df[col].fillna(0, inplace=False).astype(dtype)
        elif pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].fillna(0.0, inplace=False).astype(dtype)
        elif pd.api.types.is_bool_dtype(dtype):
            df[col] = df[col].fillna(False, inplace=False).astype(dtype)
        elif pd.api.types.is_object_dtype(dtype):
            df[col] = df[col].fillna('', inplace=False)  # 字符串填充空字符串
    return df

datapath = 'data/GiveMeSomeCredit/'

# # EDA for sampleEntry.csv
# # 读取 CSV 文件
# testDataProb = pd.read_csv(datapath + 'sampleEntry.csv')
# # 显示前几行数据
# print(f"\ntestDataProb(head):\n{testDataProb.head()}\n")

# EDA for cs-training.csv
trainData = pd.read_csv(datapath + 'cs-training.csv')
trainData = trainData.drop(columns=['Unnamed: 0'])  # 删除第一列
# 填充缺失值
trainData = fill_na_with_default(trainData)
# 显示前几行数据
# print(f"\ntrainData(head):\n{trainData.head()}\n")
# 显示每列的最大最小值
print("Max\n", trainData.max())
print("Min\n", trainData.min())
print(f"Information of trainData:\n{trainData.info()}\n")
trainData.to_csv("data/ProcessedData/cs-training-processed.csv", index=False)

# EDA for cs-test.csv
testData = pd.read_csv(datapath + 'cs-test.csv')
testData = testData.drop(columns=['Unnamed: 0'])  # 删除第一列
# 填充缺失值

# 显示每列的最大最小值
print("Max\n", testData.max())
print("Min\n", testData.min())
print(f"Information of testData:\n{testData.info()}\n")
testData.to_csv("data/ProcessedData/cs-test-processed.csv", index=False)