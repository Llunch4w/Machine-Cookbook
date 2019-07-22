from sklearn import preprocessing as prework

# 将训练集中某一列数值特征（假设是第i列）的值缩放成均值为0，方差为1的状态
def mean_removal(data):
    data_std = prework.scale(data)
    return data_std

# 将特征的数值范围缩放到合理的大小,frange是缩放到的范围
def scaling(data,frange):
    data_scaler = prework.MinMaxScaler(feature_range=frange)
    data_scaled = data_scaler.fit_transform(data)
    return data_scaled

# 机器学习中最常用的归一化形式就是将特征向量调整为L1范数，使特征向量的数值之和为1
def normalization(data):
    data_normalized = prework.normalize(data,norm='l1')
    return data_normalized

# 二值化用于将数值特征向量转化为布尔类型特征向量
def binarization(data):
    data_binarizer = prework.Binarizer(threshold=1.4)
    data_bin = data_binarizer.transform(data)
    return data_bin

