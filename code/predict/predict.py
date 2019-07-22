from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def plot_classifier(classifier,x,y,title,annotate=False):
    # 定义图形的取值范围
    x_min,x_max = min(x[:,0])-1.0,max(x[:,0])+1.0
    y_min,y_max = min(x[:,1])-1.0,max(x[:,1])+1.0
    # 预测值表示我们在图形中想要使用的数值范围，我们增加了一些余量，例如上述代码中的1.0
    
    # 为了画出边界，还需要利用一组网格数据求出方程的值，然后把边界画出来
    step_size = 0.01 # 网格步长 
    x_values,y_values = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size)) # 定义网格
    
    # 计算出分类器对所有数据点的分类结果
    mesh_output = classifier.predict(np.c_[x_values.ravel(),y_values.ravel()]) # 分类器输出结果
    mesh_output = mesh_output.reshape(x_values.shape) # 数组维度变形
    ## np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
    ## np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    ## ravel()将多维数组降位一维
    
    # 用彩色区域画出各个类型的边界
    plt.figure()
    plt.pcolormesh(x_values,y_values,mesh_output,cmap=plt.cm.gray) # 选择配色方案
    
    # 接下来再把训练数据点画在图上
    plt.scatter(x[:,0],x[:,1],c=y,s=30,edgecolors='blue',linewidth=1,cmap=plt.cm.Paired)
    # c=y表示颜色使用顺序（y对应的是label），用目标标记映射cmap的颜色表
    plt.xlim(x_values.min(),x_values.max())
    plt.ylim(y_values.min(),y_values.max())
    
    plt.xticks(np.arange(int(x_min),int(x_max),1.0))
    plt.yticks(np.arange(int(y_min),int(y_max),1.0))
    
    plt.title(title)

    if annotate:
        for a,b in zip(x[:, 0], x[:, 1]):
            # Full documentation of the function available here: 
            # http://matplotlib.org/api/text_api.html#matplotlib.text.Annotation
            plt.annotate(
                '(' + str(round(a, 1)) + ',' + str(round(b, 1)) + ')',
                xy = (a, b), xytext = (-15, 15), 
                textcoords = 'offset points', 
                horizontalalignment = 'right', 
                verticalalignment = 'bottom', 
                bbox = dict(boxstyle = 'round,pad=0.6', fc = 'white', alpha = 0.8),
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
    
    plt.show()



def svm(params,x_data,y_data):
    from sklearn.model_selection import train_test_split
    # params = {'kernel':'rbf','probability':True,'class_weight':'balanced'}

    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.25,random_state=5)
    classifier = SVC(**params)
    classifier.fit(x_train,y_train)

    # 训练集合分类结果
    plot_classifier(classifier,x_train,y_train,'Train dataset')

    target_names = ['Class-' + str(int(i)) for i in set(y_data)]
    print('\n','#'*30)
    print('\nClassifier performance on training dataset\n')
    y_pred = classifier.predict(x_train)
    print('\n',classification_report(y_train,y_pred,target_names=target_names),'\n')
    print('\n','#'*30)

    # 测试集合分类结果
    plot_classifier(classifier,x_test,y_test,'Test dataset')

    target_names = ['Class-' + str(int(i)) for i in set(y_data)]
    print('\n','#'*30)
    print('\nClassifier performance on testing dataset\n')
    y_test_pred = classifier.predict(x_test)
    print('\n',classification_report(y_test,y_test_pred,target_names=target_names),'\n')
    print('\n','#'*30)

    return classifier

# svm用作回归器
def svm2(params,x_data,y_data):
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as sm

    x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.25,random_state=5)
    regressor = SVC(**params)
    regressor.fit(x_train,y_train)
    
    y_test_pred = regressor.predict(x_test)
    print('均方误差 = ',round(sm.mean_squared_error(y_test,y_test_pred),2))
    
    return regressor



def confidence(classifier,points):
    print('\nConfidence measure:')
    for i in input_datapoints:
        print(i,'-->',classifier.predict_proba(i.reshape(1,2))[0])


def optimal(metrics,parameter_grid):
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm

    for metric in metrics:
        print('\n### Searching optimal hyperparameters for',metric)
        classifier = GridSearchCV(svm.SVC(C=1),parameter_grid,cv=5,scoring=metric)
        classifier.fit(x_train,y_train)
        print('\nScores across the parameter grid:')
        
        params_group =  classifier.cv_results_['params']
        score_group =  classifier.cv_results_['mean_test_score']
        for i in range(len(params_group)):
            print(params_group[i],'-->',round(score_group[i],3))
        
        print('\nHighest Scoring parameter set:',classifier.best_params_)

    # # 通过交叉检验设置参数
    # parameter_grid = [
    #     {'kernel':['linear'],'C':[1,10,50,600]},
    #     {'kernel':['poly'],'degree':[2,3]},
    #     {'kernel':['rbf'],'gamma':[0.01,0.001],'C':[1,10,50,600]}
    # ]

    # # 定义需要使用的指标
    # metrics = ['precision','recall_weighted']
    

    
