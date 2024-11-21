from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit
from PySide2.QtWidgets import QFileDialog
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix
from PySide2.QtUiTools import QUiLoader
from sklearn.linear_model import LinearRegression,BayesianRidge,ElasticNet
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.linear_model import Ridge
import configparser
import os.path
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
import sys


warnings.filterwarnings("ignore")
# 全局，错误数据的索引
globals_dict = {}
globals_right = {}
kfold=KFold(n_splits=10)
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows',None)



def getConfig(section,key=None):
    config = configparser.ConfigParser()  #初始化一个configparser类对象
    dir = os.path.dirname(os.path.abspath(__file__)) #获取当前文件的文件夹位置
    #file_path = dir+'\\config\\config.ini'  #完整的config.ini文件路径
    config.read(os.path.join(dir,'config.ini'))
    #config.read(file_path,encoding='utf-8') #读取config.ini文件内容
    if key!=None:
        return config.get(section,key)  #获取某个section下面的某个key的值
    else:
        return config.items(section)  #或者某个section下面的所有值





class classification:
    def logistic_regression_test(self):
        gc = LinearRegression()
        gc.fit(x_train, y_train)
        y_predict = gc.predict(x_test)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)
        #acc = accuracy_score(y_predict, y_test)
        print(y_predict)
        #score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)


        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE


        return RMSE,MAE,r2_score(y_test, y_predict)


    def knn(self):
        print('knn')
        gc = KNeighborsRegressor(n_neighbors=int(getConfig('knn','n_neighbors')))
        gc.fit(x_train, y_train)
        y_predict = gc.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)
        print(y_test)
        print("knn准确率：", gc.score(x_test, y_test))
        # print(classification_report(y_test,y_predict,labels=[0,1],target_names=["不超标","超标"]))
        print(y_predict)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)
        #acc = accuracy_score(y_predict, y_test)

        #score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)


        error = []
        print(y_test.values,y_predict)


        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        return RMSE,MAE,r2_score(y_test, y_predict)

    def decision_tree(self):
        print('决策树')
        gc=DecisionTreeRegressor()
        gc.fit(x_train, y_train)

        y_predict = gc.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)
        print("逻辑回归准确率：", gc.score(x_test, y_test))
        # print(classification_report(y_test,y_predict,labels=[0,1],target_names=["不超标","超标"]))
        print(y_predict)

        error = []
        print(y_test.values,y_predict)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)
        #acc = accuracy_score(y_predict, y_test)

        #score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)


        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        return RMSE,MAE,r2_score(y_test, y_predict)


    def random_forest(self):
        print('随即森马')

        # 交叉验证与网格搜索
        gc = RandomForestRegressor(n_estimators=int(getConfig('random_forest','n_estimators')))
        gc.fit(x_train, y_train)

        y_predict = gc.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)


        #min_max_scaler = preprocessing.MinMaxScaler()
        #y_predict = min_max_scaler.fit_transform(y_predict)
        error = []
        print(y_test.values,y_predict)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)
        #acc = accuracy_score(y_predict, y_test)
        print(y_predict)
        #score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)


        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        return RMSE,MAE,r2_score(y_test, y_predict)
    def nn(self):
        print('神经网络')

        # 交叉验证与网格搜索
        gc = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
        gc.fit(x_train, y_train)

        y_predict = gc.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)


        #min_max_scaler = preprocessing.MinMaxScaler()
        #y_predict = min_max_scaler.fit_transform(y_predict)
        error = []
        print(y_test.values,y_predict)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)
        #acc = accuracy_score(y_predict, y_test)
        print(y_predict)
        #score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)


        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        return RMSE,MAE,r2_score(y_test, y_predict)

    def LightGBM(self):
        print('svm')

        gc = lgb.LGBMRegressor(learning_rate = float(getConfig('LightGBM','learning_rate')), max_depth = int(getConfig('LightGBM','max_depth')),n_estimators = int(getConfig('LightGBM','n_estimators')),boosting_type=str(getConfig('LightGBM','boosting_type')),random_state=int(getConfig('LightGBM','random_state')),objective=str(getConfig('LightGBM','objective')))

        gc.fit(x_train, y_train,eval_metric=str(getConfig('LightGBM','eval_metric')), verbose=int(getConfig('LightGBM','verbose')))

        y_predict = gc.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)
        #acc = accuracy_score(y_predict, y_test)
        print(y_predict)
        #score = mean_squared_error(y_predict, y_test)

        print(y_test.values,y_predict)

        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        return RMSE,MAE,r2_score(y_test, y_predict)

    def mountain(self):
        print('nn')

            # 使用岭回归
        ridge_383 = Ridge(alpha=int(getConfig('mountain','alpha')))
        ridge_383.fit(x_train, y_train)

        y_predict = ridge_383.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)

        print(y_predict)
        #score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)

        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        return RMSE,MAE,r2_score(y_test, y_predict)


    def BayesianRidge(self):

        br_383 =BayesianRidge()
        br_383.fit(x_train, y_train)

        br_383.fit(x_train, y_train)
        #y_predict = br_383.predict(x_test)
        y_predict = br_383.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)
        error = []
        print(y_test.values,y_predict)

        #y_test_pred = (y_predict - y_predict.min()) / (y_predict.max() - y_predict.min())
        #y_test_pred[y_test_pred > 0.5] = 1
        #y_test_pred[y_test_pred <= 0.5] = 0
        #acc = accuracy_score(y_test_pred, y_test)
        #acc = accuracy_score(y_predict, y_test)
        print(y_predict)
        #score = mean_squared_error(y_predict, y_test)

        print(y_test.values,y_predict)

        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        print("Errors: ", error)
        print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        print("Square Error: ", squaredError)
        print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE=sqrt(sum(squaredError) / len(squaredError))
        MAE=sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        return RMSE,MAE,r2_score(y_test, y_predict)


class Stats:

    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('回归.ui')

        #self.ui.chooseButton.clicked.connect(self.handleCalc)

        self.ui.comboBox.currentIndexChanged.connect(self.combox)

        self.ui.button.clicked.connect(self.choose_file)

        self.ui.col_button.clicked.connect(self.get_col)

        self.ui.lab_button.clicked.connect(self.get_label)
        self.ui.start_button.clicked.connect(self.get_data)
        self.ui.preview_button.clicked.connect(self.previwe)
        self.ui.train_button.clicked.connect(self.train_data)
        self.ui.val_button.clicked.connect(self.val_data)
        self.ui.result_button.clicked.connect(self.result)
        self.ui.train_out.clicked.connect(self.train_out)
        self.ui.test_out.clicked.connect(self.test_out)



    def choose_file(self):

        filePath, _ = QFileDialog.getOpenFileName(self.ui)
        self.ui.text.setText(filePath)
        #filePath = QFileDialog.getExistingDirectory(self.ui, "选择存储路径")
        text = self.ui.text.toPlainText()
        print(filePath)
        global df
        df = pd.read_excel(filePath)



    def combox(self):
        method = self.ui.comboBox.currentText()
        print(method)
        global RMSE,MAE,sc
        if method=='线性回归模型':
            RMSE,MAE,sc = classification().logistic_regression_test()
        if method=='k近邻回归模型':
            RMSE,MAE,sc=classification().knn()
        if method=='LightGBM':
            RMSE,MAE,sc=classification().LightGBM()
        if method=='决策树回归模型':
            RMSE,MAE,sc=classification().decision_tree()
        if method == '随机森林回归模型':
            RMSE,MAE,sc=classification().random_forest()
        if method == '普通岭回归':
            RMSE,MAE,sc=classification().mountain()
        #if method == 'ElasticNet弹性网络':
            #acc,pre,rec,f1=classification().ElasticNet()
        if method == '贝叶斯岭回归':
            RMSE,MAE,sc=classification().BayesianRidge()
        if method == 'all':
            global RMSE_l,MAE_l,sc_l,RMSE_k,MAE_k,sc_k,RMSE_n,MAE_n,sc_n,RMSE_lig,MAE_lig,sc_lig,RMSE_r,MAE_r,sc_r,RMSE_rad,MAE_rad,sc_rad,RMSE_mou,MAE_mou,sc_mou,RMSE_bayes,MAE_bayes,sc_bayes

            RMSE_l,MAE_l,sc_l = classification().logistic_regression_test()
            RMSE_k,MAE_k,sc_k = classification().knn()
            RMSE_n, MAE_n, sc_n = classification().nn()
            RMSE_lig,MAE_lig,sc_lig = classification().LightGBM()
            RMSE_r,MAE_r,sc_r = classification().decision_tree()
            RMSE_rad,MAE_rad,sc_rad = classification().random_forest()
            RMSE_mou,MAE_mou,sc_mou =classification().mountain()
            RMSE_bayes,MAE_bayes,sc_bayes = classification().BayesianRidge()



    def get_col(self):
        global col1
        col1 = self.ui.col_name.toPlainText()
        self.ui.col_name.clear()


    def get_label(self):
        global label1
        label1 = self.ui.label_name.toPlainText()
        #print(label1)
        self.ui.label_name.clear()




    def get_data(self):

        col = col1.split(',')
        label = label1.split(',')
        #print(df)
        #print(col)
        global X,Y
        Y = df[label]
        #print(label)


        #print(Y)
        # X=df.loc[:, ['Pht','CEC_log','SOM_log','Cd_log']]
        # XY=df.loc[:, ['Pht','CEC_log','SOM_log','Cd_log','label']]
        X = df.loc[:, col]
        col_all=col.copy()
        col_all.extend(label)
        #print(col_all)

        global x_train, x_test, y_train, y_test,XY

        XY = df.loc[:, col_all]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=True)
        #print(X,XY)
        global df_val,y_val,x_val
        df_val = df
        #print(x_train, x_test, y_train, y_test)
        print('------------------------------------------------------------------------------')
        y_val = df_val[label]
        x_val = df_val.loc[:, col]
        # x_test = x_val
        # y_test = y_val


    def previwe(self):
        text=str(XY)
        self.ui.preview_window.setText(text)

    def train_data(self):
        global all_tr
        all_tr = pd.concat([x_train,y_train],axis=1)
        text = str(all_tr)
        self.ui.train_data.setText(text)

    def val_data(self):
        global all_te
        all_te=pd.concat([x_test,y_test],axis=1)
        text=str(all_te)
        self.ui.val_data.setText(text)

    def train_out(self):
        pd.DataFrame(all_tr)
        all_tr.to_excel('训练集数据.xlsx')

    def test_out(self):
        pd.DataFrame(all_te)
        all_te.to_excel('测试集数据.xlsx')

    def result(self):
        method = self.ui.comboBox.currentText()
        if method !='all':
            text='回归模型:{method}\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE, MAE= MAE, sc=sc)
            self.ui.result.setText(text)
        if method == 'all':
            text_1 = '回归模型:线性回归\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_l, MAE= MAE_l, sc=sc_l)
            text_2 = '回归模型:k近邻\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_k, MAE= MAE_k, sc=sc_k)
            text_7 = '回归模型:神经网络\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_n, MAE= MAE_n, sc=sc_n)
            text_3 = '回归模型:LightGBM\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_lig, MAE= MAE_lig, sc=sc_lig)
            text_4 = '回归模型:决策树\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_r, MAE= MAE_r, sc=sc_r)
            text_5 = '回归模型:随机森林\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_rad, MAE= MAE_rad, sc=sc_rad)
            text_6='回归模型:岭回归\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_mou, MAE= MAE_mou, sc=sc_mou)

            text_8 = '回归模型:贝叶斯\nRMSE:{RMSE}\nMAE:{MAE}\n准确率:{sc}\n'.format(method=method, RMSE=RMSE_bayes, MAE= MAE_bayes, sc=sc_bayes)

            self.ui.result.setText(text_1+text_2+text_7+text_3+text_4+text_5+text_6+text_8)



app = QApplication([])
stats = Stats()
stats.ui.show()



app.exec_()








