import pandas as pd   #导入pandas库
import numpy as np
import matplotlib.pyplot as plt
#导入需要的模块
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import plot_tree
from pylab import mpl
from sklearn.ensemble import RandomForestRegressor
# 使用交叉验证
from sklearn.model_selection import KFold
# 逐步回归
from statsmodels.formula.api import ols #加载ols模型
import re

# 设置中文显示字体 #图画
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

#label = label1

#models.replace("NaN",np.nan,inplace=True)
#models=models.dropna(how="any")

# 预处理
def proce(all_data, all_column):
    data_to_normalize = all_data[all_column].values
    min_max_scaler = preprocessing.MinMaxScaler()
    all_data[all_column] = min_max_scaler.fit_transform(data_to_normalize)
    normalized_df = pd.DataFrame(all_data, columns=all_column)
    all_data[all_column] = normalized_df
    all_data = all_data.round(3)
    return all_data

class Project2:
    def log(self,col,label):
        print('------------------------开始执行逐步回归---------------------------------')
        # 处理特殊符号
        col1 = []
        for i in all_data.columns:
            i = re.sub(u"\\（.*?\\）|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", i)
            col1.append(i)
        all_data.columns = col1


        col1 = []
        for i in col:
            i = ''.join(i)
            i = re.sub(u"\\（.*?\\）|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", i)
            col1.append(i)
        col = col1

        #data = all_data.loc[:, col]  # 对应自变量数据
        # print(data)

        temp = []
        #label = label1.split(',')
        label = ''.join(label)
        label = re.sub(u"\\（.*?\\）|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", label)

        # print(label)
        temp.append(label)

        col_all = col.copy()
        col_all.extend(temp)

        # 完整训练数据集
        data_all = all_data.loc[:, col_all]
        # print(data_all)

        # 逐步回归过程
        data = data_all
        target = label
        variate = set(data.columns)  # 将字段名转换成字典类型
        print(variate)
        variate.remove("".join(target))  # 去掉因变量的字段名
        print(variate)
        selected = []
        current_score, best_new_score = float('inf'), float('inf')  # 目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
        # 循环筛选变量
        while variate:
            aic_with_variate = []
            for candidate in variate:  # 逐个遍历自变量
                formula = "{}~{}".format(target, "+".join(selected + [candidate]))  # 将自变量名连接起来
                aic = ols(formula=formula, data=data).fit().aic  # 利用ols训练模型得出aic值
                aic_with_variate.append((aic, candidate))  # 将每一次的aic值放进空列表
                print('自变量为{}，对应的AIC值为：{}'.format("+".join(selected + [candidate]), aic))
                # print(aic_with_variate)
            aic_with_variate.sort(reverse=True)  # 降序排序aic值
            best_new_score, best_candidate = aic_with_variate.pop()  # 最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
            if current_score > best_new_score:  # 如果目前的aic值大于最好的aic值
                variate.remove(best_candidate)  # 移除加进来的变量名，即第二次循环时，不考虑此自变量了
                selected.append(best_candidate)  # 将此自变量作为加进模型中的自变量
                current_score = best_new_score  # 最新的分数等于最好的分数
                print("最小AIC值为：{}".format(current_score))  # 输出最小的aic值
            else:
                print("for selection over!")
                break
        formula = "{}~{}".format(target, "+".join(selected))  # 最终的模型式子
        print("final formula is {}".format(formula))
        print("主因为：{}".format(",".join(selected)))
        print(selected)
        model = ols(formula=formula, data=data).fit()
        return selected


    def Dtree(self,col,label):
        print('------------------------开始执行决策树---------------------------------')
        label = np.asarray(label)
        # print(df)
        #global X, Y
        Y = data_y
        # print(label)
        # print(Y)
        X = data_x

        col_all = col.copy()
        np.concatenate([col_all,label],axis=0)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
        # print(x_train)
        # global df_val, y_val, x_val
        # df_val = all_data
        # print(x_train, x_test, y_train, y_test)

        # y_val = df_val[label]
        # x_val = df_val.loc[:, col]

        model = RandomForestRegressor() #改1（gc改为model）
        fold_accuracy = []
        kfold = KFold(n_splits=5)

        for train_index, test_index in kfold.split(x_train, y_train):
            # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标  X_train.iloc[train_index,:]
            this_train_x, this_train_y = x_train.iloc[train_index, :], y_train.iloc[train_index, :]  # 本组训练集
            this_test_x, this_test_y = x_train.iloc[test_index, :], y_train.iloc[test_index, :]  # 本组验证集
            # 训练本组的数据，并计算准确率
            this_train_y = this_train_y.values.ravel() #改1 （添加）
            model.fit(this_train_x, this_train_y)
            prediction = model.predict(this_test_x)
            valid_acc = model.score(this_test_x, this_test_y)
            fold_accuracy.append(valid_acc)
            # print(confusion_matrix(y_test, gc.predict(this_test_x)))

            # print(prediction)
        # print(this_test_y)

        # score = gc.score(x_test, y_test)

        print("Accuracy per fold: ", fold_accuracy, "\n")
        print("Average accuracy: ", sum(fold_accuracy) / len(fold_accuracy))
        # gc.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        # score = mean_squared_error(y_predict, y_test)
        print("算法准确率:", model.score(x_test, y_test))
        # print(classification_report(y_test,y_predict,labels=[0,1],target_names=["不超标","超标"]))
        # print(y_predict)

        error = []
        # print(y_test.values, y_predict)

        for i in range(len(y_test)):
            error.append(y_test.values[i] - y_predict[i])

        # print("Errors: ", error)
        # print(error)

        squaredError = []
        absError = []
        for val in error:
            squaredError.append(val * val)  # target-prediction之差平方
            absError.append(abs(val))  # 误差绝对值

        # print("Square Error: ", squaredError)
        # print("Absolute Value of Error: ", absError)

        # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
        RMSE = np.sqrt(sum(squaredError) / len(squaredError))
        MAE = sum(absError) / len(absError)

        print("RMSE = ", RMSE)  # 均方根误差RMSE
        print("MAE = ", MAE)  # 平均绝对误差MAE

        print("随机森林中决策树的数量：", model.n_estimators) #改1
        gc = model.estimators_[0]
        mse_values = gc.tree_.impurity

        print("叶节点的平方误差值：", mse_values)

        print(gc.feature_importances_)


        DT_val=gc.feature_importances_
        DT=pd.DataFrame(DT_val,index=col)
        #print(DT)
        DT=DT.T
        DT = DT.sort_values(by=0, axis=1, ascending=False)  # ascending表示是否升序,axis=1表示对行操作
        DT.rename(index={0: '重要性'}, inplace=True)
        print(DT.T)
        for i in DT.columns:
            if DT.loc['重要性'][i] == 0:
                DT.drop(columns=i, inplace=True)
        print(DT.columns.values)
        DT_end=DT.columns.values
        col_dt = []
        for i in DT_end:
            i = re.sub(u"\\（.*?\\）|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", i)
            col_dt.append(i)
        #print(col_dt)
        #t = tree.export_text(gc, feature_names=list(col))  决策树文字版
        #print(t)

        plt.figure(figsize=(15, 9))
        plot_tree(gc, filled=True, feature_names=list(col))
        plt.show()
        #feature_name = col
        #dot_data = tree.export_graphviz(gc,feature_names=feature_name,class_names=['琴酒','雪莉','贝尔摩德'],filled=True,rounded=True)# 圆角
        #graph = graphviz.Source(dot_data)
        #graph.save('./tree.dot') #保存决策树
        #dot - Tpng F:\system\桌面\pythonProject\2023-2-4准备\tree.dot - o F:\system\桌面\pythonProject\2023-2-4准备\tree.png
        return col_dt

if __name__ == '__main__':

    excel_file_x = './实验数据x.xlsx'  # 导入excel数据(x
    excel_file_y = './实验数据y.xlsx'  # 导入excel数据(y

    x_data = pd.read_excel(excel_file_x)  # 读取excel
    y_data = pd.read_excel(excel_file_y)  # 读取excel

    col = ['pH','有机质（g/kg）','全氮（mg/kg）','有效磷（mg/kg）','速效钾（mg/kg）','缓效钾（mg/kg）','有效锌（mg/kg）',
           '有效硼（mg/kg）','有效钼（mg/kg）','有效铜（mg/kg）','有效硅（mg/kg）','有效锰（mg/kg）','有效铁（mg/kg）']    ##x的列名（修改这里就行）

    label = ['可溶性固形物级别（数值越大品质越差）']  ##y的列名（修改这里就行）
                                                       ##因为之后的函数需要col和label这两个参数，所以提前单独写出来

    all_data = pd.concat([x_data, y_data], axis=1)
    #print(all_data)
    all_data = proce(all_data,col+label)  # 预处理
    #all_data = proce(x_data, col)

    data_x = all_data.loc[:, col]
    data_y = all_data.loc[:, label]

    p2 = Project2()
    Dtree = p2.Dtree(col,label)  # 随机森林
    log = p2.log(col,label)  # 逐步回归

    end_2 = np.union1d(Dtree, log) #取并集
