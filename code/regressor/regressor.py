import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,explained_variance_score

def data_split(x_data,y_data,size=0.2):
    # 打乱顺序
    x_data,y_data = shuffle(x_data,y_data,random_state=7)   
    # 根据size划分数据集
    num_training = int((1-size) * len(x_data))
    x_train,y_train = x_data[:num_training],y_data[:num_training]
    x_test,y_test = x_data[num_training:],y_data[num_training:]
    return x_train,x_test,y_train,y_test

def plot(x_train,y_train,y_train_predict):
    plt.figure()
    plt.scatter(x_train,y_train,color='g')
    plt.plot(x_train,y_train_predict,color='black',linewidth=4)
    plt.title('data')
    plt.show()

def linear(x_train,y_train,x_test,y_test):
    linear_regressor = linear_model.LinearRegression()
    linear_regressor.fit(x_train,y_train)
    y_test_predict = linear_regressor.predict(x_test)
    estimate(y_test,y_test_predict)
    return linear_regressor

def ridge(x_train,y_train,x_test,y_test):
    ridge_regressor = linear_model.Ridge(alpha=1,fit_intercept=True,max_iter=10000)
    ridge_regressor.fit(x_train,y_train)
    y_test_predict = ridge_regressor.predict(x_test)
    estimate(y_test,y_test_predict)
    return ridge_regressor

def poly(x_train,y_train,x_test,y_test,degree):
    polynomial = PolynomialFeatures(degree=degree)
    x_train_transformed = polynomial.fit_transform(x_train)
    poly_linear_model = linear_model.LinearRegression()
    poly_linear_model.fit(x_train_transformed, y_train)
    x_test_transformed = polynomial.fit_transform(x_test)
    y_test_predict = poly_linear_model.predict(x_test_transformed)
    estimate(y_test,y_test_predict)
    # datapoint = np.array([0.39,2.78,7.11]).reshape((1,3))
    # poly_datapoint = polynomial.fit_transform(datapoint)
    return poly_linear_model

def deciTree(x_train,y_train,x_test,y_test):
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(x_train,y_train)
    y_test_predict = dt_regressor.predict(x_test)
    estimate(y_test,y_test_predict)
    return dt_regressor

def adaBoost(x_train,y_train,x_test,y_test):
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
    ab_regressor.fit(x_train,y_train)
    y_test_predict = ab_regressor.predict(x_test)
    estimate(y_test,y_test_predict)
    return ab_regressor

def randForest(x_train,y_train,x_test,y_test):
    rf_regressor = RandomForestRegressor(n_estimators=1000,max_depth=10)
    rf_regressor.fit(x_train,y_train)
    y_test_predict = rf_regressor.predict(x_test)
    estimate(y_test,y_test_predict)
    return rf_regressor

def plot_feature_importances(feature_importances,title,feature_names):
    # 将重要性能标准化
    feature_importances = 100.0 * (feature_importances/max(feature_importances))  
    # 将得分从高到低排序
    # flipud是为了倒序
    index_sorted = np.flipud(np.argsort(feature_importances))   
    # 让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5   
    #画条形图
    plt.figure()
    plt.bar(pos,feature_importances[index_sorted],align='center')
    plt.xticks(pos,feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

def estimate(y_test,y_test_predict):
    print('均方误差 = ',round(mean_squared_error(y_test,y_test_predict),2))
    print('解释方差分 = ',round(explained_variance_score(y_test,y_test_predict),2))

def save(model,filename):
    import pickle
    output_model_file = filename # 'saved_model.pkl'
    with open(output_model_file,'wb') as f:
        pickle.dump(model,f)

def load(filename):
    with open(filename,'rb') as f:
        model = pickle.load(f)
    return model



