


# import pandas as pd
#
# # 读取CSV文件
# df = pd.read_csv('music_genre_mturk.csv')
#
# # 提取id字段的第一个句号之前的部分作为true_label
# df['true_label'] = df['id'].apply(lambda x: x.split('.')[0])
#
# # 将处理后的DataFrame保存到新的CSV文件中
# df.to_csv('new.csv', index=False)
#
# print("CSV文件处理完成，并已保存到new.csv文件中。")
##############################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取 new.csv 文件
df = pd.read_csv('../new.csv')

# 1. 生成 data_train.npy
# 提取 feature0 到 feature123 列的数据
data_train = df.iloc[:, 2:126].values  # 第2列到第125列是 feature0 到 feature123
# 保存为 data_train.npy
np.save('data_train.npy', data_train)

# 2. 生成 answers.npy
# 获取唯一的 annotator 列表
annotators = df['annotator'].unique()
num_annotators = len(annotators)
num_instances = len(df['id'])

# 初始化 answers 数组，形状为 (2945, num_annotators)，初始值为 -1
answers = -1 * np.ones((num_instances, num_annotators))

# 创建一个字典，将 annotator 映射到 answers 数组的列索引
annotator_to_index = {annotator: idx for idx, annotator in enumerate(annotators)}

# 使用 LabelEncoder 将字符串标签转换为整数
label_encoder = LabelEncoder()
df['class_encoded'] = label_encoder.fit_transform(df['class'])  # 对 class 列进行编码

# 输出标签编码器的映射关系
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("标签映射关系:", label_mapping)

# 填充 answers 数组
for idx, row in df.iterrows():
    annotator_idx = annotator_to_index[row['annotator']]
    class_label_encoded = row['class_encoded']  # 获取编码后的标签
    answers[idx, annotator_idx] = class_label_encoded  # 将编码后的标签赋值给 answers 数组

answers = answers.astype(int)
# 保存为 answers.npy
np.save('answers.npy', answers)

# 3. 生成 labels_train.npy
# 提取 true_label 列的数据
true_labels = df['true_label']  # 每个实例的真实标签

# 使用相同的 LabelEncoder 对 true_label 进行编码
true_labels_encoded = label_encoder.transform(true_labels)

# 保存为 labels_train.npy
np.save('labels_train.npy', true_labels_encoded)

print("文件生成完成！")

# 验证保存的文件是否正确
loaded_features = np.load('data_train.npy')
loaded_answers = np.load('answers.npy')
loaded_labels = np.load('labels_train.npy')

print("data_train.npy 的形状:", loaded_features.shape)  # 应为 (2945, 124)
print(loaded_answers)
# print("answers.npy 的形状:", loaded_answers.shape)  # 应为 (2945, num_annotators)
# print("labels_train.npy 的形状:", loaded_labels.shape)  # 应为 (2945,)