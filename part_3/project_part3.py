import configparser
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore") # 忽略警告

all_filepath = './果品品质提升案例（0307）.xlsx'
all_df = pd.read_excel(all_filepath)


# 计算强度
def add_rf(key, file_path,test_name):
    # 读取数据，训练随机森林模型，存储后返回模型名称（文件存储名称）
    excel_file = file_path  # 导入excel数据
    df = pd.read_excel(excel_file)
    # 输出数据预览
    print(df.head())
    # 自变量
    x = 'TpH,T有机质（g/kg）,全氮（mg/kg）,碱解氮（mg/kg）,有效磷（mg/kg）,速效钾（mg/kg）,缓效钾（mg/kg）,有效锌（mg/kg）,有效硼（mg/kg）,有效钼（mg/kg）,有效铜（mg/kg）,有效硅（mg/kg）,有效锰（mg/kg）,有效铁（mg/kg）,有效硫（mg/kg）,交换性钙（mg/kg）,交换性镁（mg/kg）,T孔隙度%,T容重g/cm3,T含水量%'
    # 因变量
    y = '施用强度(kg/亩)'

    col1 = x
    label = y
    col = col1.split(',')
    # print(col)
    col = np.hstack([col,test_name])
    # print(col)
    label = label.split(',')
    label = np.asarray(label)
    # print(df)

    Y = df[label]
    # print(label)

    X = df.loc[:, col]
    # print(X)
    # print(Y)
    col_all = col.copy()
    np.concatenate([col_all, label], axis=0)


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    print('随机森林训练生成施肥强度的数据路径为', file_path)
    model_name = key + '.joblib'

    joblib.dump(forest, model_name)  # 保存模型


    return model_name

# 计算成本
def add_logic(key, file_path):
    # 读取数据，训练回归，存储后返回模型名称（文件存储名称）
    print('逻辑回归训练生成施肥成本的数据路径为', file_path)
    model_name = key + '.joblib'

    excel_file = file_path  # 导入excel数据
    df = pd.read_excel(excel_file)

    label_x = ['施用强度(kg/亩)']
    label_y = ['治理成本（元/亩）']

    XX = df[label_x]
    YY = df[label_y]

    X_train2, X_test2, y_train2, y_test2 = train_test_split(XX, YY, test_size=0.2, random_state=1)

    # 训练线性回归模型
    lin_reg = LinearRegression()
    lin_reg.fit(X_train2, y_train2)

    # 进行预测
    y_train_pred2 = lin_reg.predict(X_train2)
    y_test_pred2 = lin_reg.predict(X_test2)

    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train2, y_train_pred2), mean_squared_error(y_test2, y_test_pred2)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train2, y_train_pred2), r2_score(y_test2, y_test_pred2)))

    joblib.dump(lin_reg, model_name)  # 保存模型

    return model_name

def change_tree(my_dict):
    while True:
        # 提示用户选择要执行的操作
        print("请选择要执行的操作：")
        print("1. 增加模型（会同时生成强度预测模型和成本预测模型）")
        print("2. 删除模型")
        print("3. 退出")

        choice = input("请输入您的选择（1/2/3）:")

        # 增加模型
        if choice == "1":
            key = input("请输入要添加的模型名称:")
            file_name = input("请输入待分析excel名称（带后缀）:")
            file_path = './'+ file_name


            test_name_1 = input("请输入期望提高的列名:")
            if test_name_1 == '单果重':
                test_name = '单果重%提升'
            if test_name_1 == '糖度':
                test_name = '糖度%提升'
            if test_name_1 == '硬度':
                test_name = '硬度（kg/cm2）5-9'
            if test_name_1 == '可滴定酸':
                test_name = '可滴定酸%0.1-0.8'
            if test_name_1 == '可溶性固形物':
                test_name = '可溶性固形物%提升'

            key1 = key + '_强度'
            key2 = key + '_成本'
            value_1 = add_rf(key1,file_path,test_name)
            value_2 = add_logic(key2,file_path)


            my_dict[key1] = value_1
            my_dict[key2] = value_2
            print("已成功添加强度计算模型，成本计算模型以及存储名称:{}:{},{}:{}".format(key1, value_1,key2, value_2))

        # 删除模型
        elif choice == "2":
            key = input("请输入要删除的模型名称:")
            if key in my_dict:
                del my_dict[key]
                print("已成功删除模型：{}".format(key))
            else:
                print("要删除的模型不存在")

        # 退出
        elif choice == "3":
            # 将字典保存到配置文件中
            config = configparser.RawConfigParser()
            config.optionxform = str  # 不进行大小写转换
            #config.optionxform = str
            config["DEFAULT"] = my_dict
            with open("my_config.ini", "w") as f:
                config.write(f)
            print("程序已退出")
            break

        # 处理无效选择
        else:
            print("无效选择，请重新输入")

def predict(my_dict):
    # 准备数据进行预测
    filepath = './data.xlsx'
    data = pd.read_excel(filepath)
    column_names = ['TpH','T有机质（g/kg）','全氮（mg/kg）','碱解氮（mg/kg）','有效磷（mg/kg）','速效钾（mg/kg）','缓效钾（mg/kg）','有效锌（mg/kg）','有效硼（mg/kg）','有效钼（mg/kg）','有效铜（mg/kg）','有效硅（mg/kg）','有效锰（mg/kg）','有效铁（mg/kg）','有效硫（mg/kg）','交换性钙（mg/kg）','交换性镁（mg/kg）','T孔隙度%','T容重g/cm3','T含水量%']
    last_name_1 = input('请输入待预测的指标:')
    if last_name_1 == '单果重':
        last_name = '单果重%提升'
    if last_name_1 == '糖度':
        last_name = '糖度%提升'
    if last_name_1 == '硬度':
        last_name = '硬度（kg/cm2）5-9'
    if last_name_1 == '可滴定酸':
        last_name = '可滴定酸%0.1-0.8'
    if last_name_1 == '可溶性固形物':
        last_name = '可溶性固形物%提升'
    # 将输入字符串解析成列表
    column_names.append(last_name)
    X = data[column_names]
    X = X.to_numpy()
    #print(X)
    X_new = np.array(X, dtype=np.float32).reshape(1, -1)
    # 将列表转换为 NumPy 数组并重新形状
    #print(X_new)
    dict_2 = {}
    dict_m = {}
    for i,j in my_dict.items():
        #print(j)
        try:
        # 加载模型
            #model = joblib.load(j)
            # 使用模型进行预测
            if '_强度' in j:
                model = joblib.load(j)
                y_pred = model.predict(X_new)
                y_pred = np.around(y_pred, decimals=2) #保留两位小数
            # 输出预测结果
                #print('模型',i)
                #print("预测结果为：", list(y_pred))
                dict_2[i.replace('_强度','')] = y_pred[0]
        except Exception:
            print('加载模型失败:'+ i)
    for i,j in my_dict.items():
        #print(j)
        try:
        # 加载模型
            #model = joblib.load(j)
            # 使用模型进行预测
            if '_成本' in j:
                model = joblib.load(j)
                data = dict_2[i.replace('_成本','')]
                data = np.array(data, dtype=np.float32).reshape(1, -1)
                y_pred2 = model.predict(data)
                y_pred2 = np.around(y_pred2, decimals=2)  #保留两位小数
            # 输出预测结果
                #print('模型',i)
                #print("预测结果为：", y_pred2)
                dict_m[i.replace('_成本','')] = y_pred2[0][0]
        except Exception:
            print('加载模型失败:'+ i)

    print('强度和成本分别为:')
    print(dict_2)
    print(dict_m)
    sorted_dict = dict(sorted(dict_m.items(), key=lambda item: item[1]))
    print('成本从低到高排序为:')
    print(sorted_dict)
    other_data = pd.DataFrame()
    for i in dict_2.keys():
        #print(i)
        try:
            index = all_df[all_df['肥料名称'] == i].index[0]   #找到对应的行号
        except Exception:
            print('找不到对应肥料:'+i)
        others = all_df.loc[index, ['技术名称','肥料名称','施用方法']]
        others['施用强度'] = dict_2[i]
        others['治理成本'] = dict_m[i]
        #print(others)
        other_data = other_data.append(others, ignore_index=True)
    #pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    other_data = other_data.sort_values('治理成本')  #按照治理成本升序排列
    # 重置索引并且不显示旧的索引列
    other_data = other_data.reset_index(drop=True)
    other_data = other_data.reindex(columns=['技术名称','肥料名称','施用强度','治理成本','施用方法'])
    print('--------------------------------------------------------')
    print('最终结果为:')
    print(other_data)



while True:
    # 读取config文件
    config = configparser.RawConfigParser()
    config.optionxform = str  # 不进行大小写转换
    config.read("my_config.ini")

    my_dict = dict(config["DEFAULT"])
    print("读取的字典为：", my_dict)
    print('现有模型为:', list(my_dict.keys()))
    print("请选择要执行的操作：")
    print("1. 增加/删除模型")
    print("2. 现有模型预测")
    print("3. 退出")
    user_choice = input("请输入您的选择（1/2/3）:")
    if user_choice == '1':
        change_tree(my_dict)
    if user_choice == '2':
        predict(my_dict)
    if user_choice == '3':
        print('程序已退出')
        exit()


