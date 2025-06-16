import numpy as np
import pandas as pd

# 标签映射关系
label_mapping = {
    'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
    'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
}

# 读取CSV文件
file_path = '../music_genre_test.csv'
data = pd.read_csv(file_path)

# 提取特征值（feature0到feature123）
features = data.iloc[:, 1:125].values  # 第1列到第124列是特征值

# 提取真实标签（最后一列）
labels = data.iloc[:, -1].values  # 最后一列是标签

# 将标签转换为数值
labels_numeric = np.array([label_mapping[label] for label in labels])

# 确保数据的形状符合要求
assert features.shape == (300, 124), "特征值的形状不正确"
assert labels_numeric.shape == (300,), "标签的形状不正确"

# 保存特征值到data_test.npy
np.save('data_test.npy', features)

# 保存标签到labels_test.npy
np.save('labels_test.npy', labels_numeric)



print("预处理完成，文件已保存为 data_test.npy 和 labels_test.npy")
# 验证保存的文件是否正确
loaded_features = np.load('data_test.npy')
loaded_labels = np.load('labels_test.npy')


print("data_test.npy 的形状:", loaded_features.shape)  # 应为 (2945, 124)
#print(loaded_features)
print("labels_test.npy 的形状:", loaded_labels.shape)  # 应为 (2945, 124)
print(loaded_labels)