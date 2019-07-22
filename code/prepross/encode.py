import numpy as np
from sklearn import preprocessing as prework

# 独热编码
def onehot_encode(data):
    encoder = prework.OneHotEncoder(categories='auto') # 不加categories这个参数的话会出现警告，原因暂时不明
    encoder.fit(data)   
    # encoded_vector = encoder.transform(raw).toarray()
    return encoder

# 标记编码
def label_encode(input_classes):
    label_encoder = prework.LabelEncoder() # 定义一个编码器
    label_encoder.fit(input_classes) # 对标签进行编码
    # encoded_labels = label_encoder.transform(labels)
    return label_encoder


