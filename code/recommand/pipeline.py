# 函数组合
import numpy as np
from functools import reduce 

def add3(input_array):
    return map(lambda x: x+3, input_array)

def mul2(input_array):
    return map(lambda x: x*2, input_array)

def sub5(input_array):
    return map(lambda x: x-5, input_array)

def function_composer(*args):
    return reduce(lambda f, g: lambda x: f(g(x)), args)
#     函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：
#     用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，
#     得到的结果再与第三个数据用 function 函数运算，最后得到一个结果


def test():
    arr = np.array([2,5,4,7])

    print("\nOperation: sub5(mul2(add3(arr)))")
    
    arr1 = add3(arr)
    arr2 = mul2(arr1)
    arr3 = sub5(arr2)
    print("Output using the lengthy way:", list(arr3))

    func_composed = function_composer(sub5, mul2, add3)
    print("Output using function composition:", list(func_composed(arr)))

    print("\nOperation: mul2(sub5(mul2(add3(sub5(arr)))))\nOutput:", \
            list(function_composer(mul2, sub5, mul2, add3, sub5)(arr)))
    
    
# 机器学习流水线
# 包括预处理、特征选择、监督学习、非监督学习等函数
from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# 生成一些示例数据
X, y = samples_generator.make_classification(
        n_informative=4, n_features=20, n_redundant=0, random_state=5)
# n_features :特征个数，n_informative：多信息特征的个数，n_redundant：冗余信息，informative特征的随机线性组合

# 特征选择器
selector_k_best = SelectKBest(f_regression, k=10)
# 随机森林分类器
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)
# 构建机器学习流水线
pipeline_classifier = Pipeline([('selector', selector_k_best), ('rf', classifier)])
# 可以选择更新这些参数
pipeline_classifier.set_params(selector__k=6, 
        rf__n_estimators=25)
# 训练分类器
pipeline_classifier.fit(X, y)
# 预测输出结果
prediction = pipeline_classifier.predict(X)
print("\nPredictions:\n", prediction)
# 打印分类器得分
print("\nScore:", pipeline_classifier.score(X, y))

# 打印被分类器选中的特征
features_status = pipeline_classifier.named_steps['selector'].get_support()
selected_features = []
for count, item in enumerate(features_status):
    if item:
        selected_features.append(count)

print("\nSelected features (0-indexed):", ', '.join([str(x) for x in selected_features]))