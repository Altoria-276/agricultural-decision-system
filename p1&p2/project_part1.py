import re
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
sns.set(font='SimHei', font_scale=0.8)  # 解决Seaborn中文显示问题

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 相关分析
cor_key = 0
# cor_key = 1  #值为0时按排序选择 ，为1时按值选择（大于多少）

cor_num = 3    # 选出前多少个
cor_value = 0.2  # 选出大于多少的成分

# 方差分析
# var_key = 0
var_key = 1

var_num = 3
var_value = 0.05   # 选出p值小于多少的成分

#  熵权法
# ent_key = 0
ent_key = 1

ent_num = 3
ent_value = 0.08  # 选出权重大于多少的成分

# 预处理
def proce(all_data, x_col):
    all_column = x_col
    data_to_normalize = all_data[all_column].values
    min_max_scaler = preprocessing.MinMaxScaler()
    all_data[all_column] = min_max_scaler.fit_transform(data_to_normalize)
    normalized_df = pd.DataFrame(all_data, columns=all_column)
    all_data[all_column] = normalized_df
    all_data = all_data.round(3)
    return all_data

# 分割字符串
def splx(col,label):
    col = col.split(',')
    label = label.split(',')
    return col, label

class Project1:
    def cor(self, data_x, data_y):
        print('-------------------------相关分析-------------------------------')
        cor_y = data_y.copy()
        cor_y = proce(cor_y,label)

        col_all = col.copy()
        col_all.extend(label)
        x = pd.concat([data_x, cor_y], axis=1)
        print(col)
        result1 = np.corrcoef(x, rowvar=False)
        result1 = pd.DataFrame(result1)
        result1.columns = col_all
        result1.index = col_all
        print('相关系数矩阵为:')
        print(result1)
        print('-----------------------------------------相关系数排序后-------------------------------------------------')
        # 将DataFrame转换为长格式
        df_long = result1.unstack().reset_index(name='Value').rename(columns={'level_0': 'Column', 'level_1': 'Row'})
        # 按绝对值从大到小排序
        abs_sorted_df = df_long.iloc[df_long['Value'].abs().sort_values(ascending=False).index]
        abs_sorted_df = abs_sorted_df.loc[abs_sorted_df['Value'] < 0.99999]
        # 输出结果
        print(abs_sorted_df.to_string(index=False, header=False))

        plt.tight_layout()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
        sns.set(font='SimHei', font_scale=0.8)  # 解决Seaborn中文显示问题

        plt.subplots_adjust(left=0.023, right=0.978, top=0.961, bottom=0.415)

        sns.heatmap(result1, annot=True, vmax=1, square=True, cmap="Reds")
        plt.tight_layout()

        plt.show()


        result2 = result1.loc[label]
        result2 = result2.sort_values(by=label, axis=1, ascending=False)  # ascending表示是否升序,axis=1表示对行操作

        result2.drop(columns=label, inplace=True) #去掉和自身的相关系数
        print('因变量与其他因素的相关系数为:')
        print(result2)


        if cor_key == 0:
            result3 = result2.iloc[:, :cor_num]
        if cor_key == 1:
            result3 = result2
            for i in result3.columns:
                if result3.loc[label][i] <= cor_value:
                    result3.drop(columns=i,inplace=True)

        print('相关分析选出的成分为:')
        print(result3.columns.values)


        return result3.columns.values

    def var(self,name):
        print('-------------------------方差分析-------------------------------')
        col = name.split(',')
        data = all_data.loc[:, col] #对应数据
        # print(data)


        col_all = col.copy()
        col_all.extend(label)
        data_all = all_data.loc[:, col_all]

        result2 = []
        for item in data:
            x = data.loc[:, item]
            y = data_all.loc[:, label]
            data_use = pd.concat([x, y], axis=1)
            data_use.columns = ['Value', 'Group']

            print('{}的方差分析计算的p值为:'.format(item))
            model = ols('Value~C(Group)', data=data_use).fit()
            anova_table = anova_lm(model, typ=2).reset_index()
            p1 = anova_table.loc[0, 'PR(>F)']
            result2.append(float(p1))
            print(p1)
        factor = col
        p = result2
        result = pd.DataFrame(p, index=factor)
        result = result.T
        result = result.sort_values(by=0, axis=1, ascending=False)
        result.rename(index={0: 'p值'}, inplace=True)
        print('整合结果:')
        print(result)

        # 绘制柱状图
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
        sns.set(font='SimHei', font_scale=0.8)  # 解决Seaborn中文显示问题

        result.iloc[:, :].plot(kind="bar")
        # 画出p=0.05的基准线
        plt.axhline(y=0.05, color='r', linestyle='--')

        plt.xticks(rotation=0)

        plt.show()

        # 选出最后结果
        if var_key == 0:
            result3 = result.iloc[:, :cor_num]
        if var_key == 1:
            result3 = result
            for i in result3.columns:
                if result3.loc['p值'][i] >= var_value:
                    result3.drop(columns=i, inplace=True)

        print('方差分析选出的成分为:')
        print(result3.columns.values)

        print('-----------------------------------------方差分析排序后-------------------------------------------------')
        print(result3.T)

        # # 绘制柱状图
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        # plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
        # sns.set(font='SimHei', font_scale=0.8)  # 解决Seaborn中文显示问题
        #
        # result3.plot(kind="bar")
        #
        # plt.show()

        # 绘制图
        # plot_box(result3, label)

        return result3.columns.values


    def ent(self,data_x):
        print('-------------------------熵权法-------------------------------')
        #name = input('请输入熵权法列名称')

        data = data_x  # 读取数据
        label_need = data.keys()
        df = data[label_need]
        a = np.array(df)
        [n, m] = a.shape
        a = 1/(1+np.exp(-a))
        cs = a.sum(axis=0)  # 逐列求和
        P = 1 / cs * a  # 求特征比重矩阵
        #print(np.log(P))
        e = -(P * np.log(P)).sum(axis=0) / np.log(n)  # 计算熵值
        g = 1 - e  # 计算差异系数
        w = g / sum(g)  # 计算权重
        # print(w)
        w1=w
        weg = pd.DataFrame(w,index=col)
        weg = weg.T
        weg = weg.sort_values(by=0, axis=1, ascending=False)  # ascending表示是否升序,axis=1表示对行操作
        weg.rename(index={0: '权重'}, inplace=True)
        print('各个成分的权重为:')
        print(weg.round(3))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
        sns.set(font='SimHei', font_scale=0.8)  # 解决Seaborn中文显示问题

        #画饼状图
        # colors = sns.color_palette('bright')
        # plt.pie(w, labels=col, colors=colors, autopct='%0.0f%%')
        # plt.show()

        if ent_key == 0:
            result = (weg.iloc[:, :ent_num])
        if ent_key == 1:
            result = weg
            for i in result.columns:
                if result.loc['权重'][i] <= ent_value:
                    result.drop(columns=i, inplace=True)


        print('熵权法选出的成分为')
        print(result.columns.values)

        # print(col)
        # print(w1)

        df = pd.DataFrame(w1,index=col)
        df=df.T
        df = df.sort_values(by=0, axis=1)
        # print(df)
        v=df.values.round(3)
        # print(v[0])
        # 计算角度
        num_categories = len(col)
        theta = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

        # 创建极坐标图
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # 设置颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, len(w1)))

        # 绘制南丁格尔玫瑰图
        bars = ax.bar(theta, df.values[0], width=0.4, align='edge', alpha=0.7, color=colors)

        # 设置极坐标刻度
        ax.set_xticks(theta)
        ax.set_xticklabels(df.columns)

        # 设置极径刻度范围
        ax.set_ylim(0, max(w1) + 0.005)
        # 设置极径刻度
        ax.set_rlabel_position(90)  # 将极径刻度标签放置在极径刻度线的外侧
        # 添加标题
        ax.set_title('南丁格尔玫瑰图')

        # 显示图形
        plt.show()

        print('-----------------------------------------熵权法排序后-------------------------------------------------')
        print(result.T)

        return result.columns.values

    def find_first_index(self,arr, value):
        for i in range(len(arr)):
            if arr[i] == value:
                return i
        return -1

    def culma(self,matrix):
        n=4
        all = ['1'] * n  # 初始化长度为n的空一维数组
        i=j=k=1000
        for i in range(n):
            if matrix[0][i] > 0.05:
                all[i] = 'a'
            else:
                all[i] = 'b'
                break
        if i<n:
            j = i+1
            for j in range(j,n):
                if matrix[i][j] >0.05:
                    all[j] = 'b'
                if matrix[i][j] <= 0.05:
                    all[j] = 'c'
                    break
        if j<n:
            k = j+1
            for k in range(k,n):
                if matrix[j][k] >0.05:
                    all[k] = 'c'
                if matrix[j][k] <= 0.05:
                    all[k] = 'd'
                    break
        if k < n:
            l = k+1
            for l in range(l, n):
                if matrix[k][l] > 0.05:
                    all[l] = 'd'
                if matrix[k][l] <= 0.05:
                    all[l] = 'e'
                    break
        index = self.find_first_index(all, 'b')
        if index == 2 and matrix[1][index]=='a':
            if matrix[1][index] > 0.05:
                all[1]=all[1]+'b'
        if index == 3 and all[1]=='a' and all[2]=='a':
            if matrix[2][index] > 0.05:
                all[2]=all[2]+'b'
                if matrix[1][index] > 0.05:
                    all[1] = all[1] + 'b'
        index = self.find_first_index(all, 'c')
        if index == 3 and all[1]=='b' and all[2]=='b':
            if matrix[2][3]>0.05:
                all[2]=all[2]+'c'
        # print(all)
        return all

    def cal_p_value(self,data_use):
        data_use.columns = ['Value', 'Group']
        model = ols('Value~C(Group)', data=data_use).fit()
        anova_table = anova_lm(model, typ=2).reset_index()
        p1 = anova_table.loc[0, 'PR(>F)']
        p1 = float(p1)
        return p1

    def var_analysis(self,data_x,data_y):
        print('-------------------------方差分析-------------------------------')

        normalized_df = data_x
        # 取标签
        label_col = data_y
        data_all = pd.concat([normalized_df, label_col], axis=1)

        # 创建字典并设置初始值为False, 用于保存最后的是否选中该列的情况
        factor_dict = {column: False for column in col}
        #print(factor_dict)
        all_df1 = pd.DataFrame()
        all_df2 = pd.DataFrame()
        all_df3 = pd.DataFrame()
        all_df4 = pd.DataFrame()
        degre = {}
        for key, value in factor_dict.items():
            #print("当前分的因素是：{}".format(key))
            key_col = normalized_df[key]
            data_with_label = pd.concat([key_col, label_col], axis=1)
            data_group_by_label = data_with_label.groupby(label)

            # 创建p值矩阵
            rows = 4
            cols = 4
            p_values_matrix = [[1 if i == j else 0 for j in range(cols)] for i in range(rows)]

            # 字典：保存某成分在各个等级下的均值、方差
            factor_avg_dict = {}

            for group_num, group_data in data_group_by_label:
                #print("当前因素的等级为：{}".format(group_num))
                # print(group_data)
                mean_values = group_data.mean()
                std_values = group_data.std()

                mean_values = mean_values.values.tolist()[:-1]
                std_values = std_values.values.tolist()[:-1]

                #print("mean: {}".format(mean_values[0]))
                #print("std: {}".format(std_values[0]))

                factor_avg_dict[group_num] = {
                    'mean': mean_values[0],
                    'variance': std_values[0]
                }

            # 同一因素按照不同等级下的均值进行排序
            sorted_dict = dict(sorted(factor_avg_dict.items(), key=lambda x: x[1]['mean'], reverse=True))
            #print("factor_avg_list dic: {}".format(factor_avg_dict))
            df_1 = pd.DataFrame(factor_avg_dict[1], index=[key+'_1'])
            df_2 = pd.DataFrame(factor_avg_dict[2], index=[key+'_2'])
            df_3 = pd.DataFrame(factor_avg_dict[3], index=[key+'_3'])
            df_4 = pd.DataFrame(factor_avg_dict[4], index=[key+'_4'])
            all_df1 = pd.concat([all_df1, df_1])
            all_df2 = pd.concat([all_df2, df_2])
            all_df3 = pd.concat([all_df3, df_3])
            all_df4 = pd.concat([all_df4, df_4])
            #print("sorted_list dic: {}".format(sorted_dict))

            # 将字典的键转换为列表，获取指定标签组的数据
            keys_list = list(sorted_dict.keys())
            # 拼接列并输出结果
            for i in range(len(sorted_dict)):
                for j in range(i + 1, len(sorted_dict)):
                    col_num1 = keys_list[i]
                    col_num2 = keys_list[j]
                    column1 = data_group_by_label.get_group(col_num1)
                    column2 = data_group_by_label.get_group(col_num2)
                    data_compare = pd.concat([column1, column2], axis=0)
                    # print(data_compare)
                    # 把两组数据拼接进行方差分析，生成p值矩阵
                    p = self.cal_p_value(data_compare)
                    p_values_matrix[i][j] = p

            #print(p_values_matrix)

            ################################################################
            # 调用标签评级函数，输入p值矩阵，返回评级列表
            label_list = self.culma(p_values_matrix)
            ################################################################

            # 返回标签之后的处理逻辑:1)将标签对应到排序后的字典中  2)如果有大于'c'整个元素组置为true
            i = 0
            temp_dict = {}
            for item in keys_list:
                temp_dict[item]=label_list[i]
                i = i + 1
            elem = [0,0,0,0]
            for it, val in temp_dict.items():
                elem[int(it)-1]=val
            if(label_list[3]>='c'):
                factor_dict[key] = True
            degre[key]=elem
        keys = [key for key, value in factor_dict.items() if value == True]

        #print(all_df1)
        #print(all_df2)
        #print(all_df3)
        #print(all_df4)
        hight = []
        low = []
        fig, ax = plt.subplots()
        for j in range(1, 5):
            for i in eval('all_df'+str(j)).index:
                low.append(eval('all_df'+str(j)).loc[i, 'mean'])
                hight.append(eval('all_df'+str(j)).loc[i, 'variance'])
            #print(eval('all_df'+str(j))['mean'].values)
            #print(list(eval('all_df'+str(j))['mean'].values))
            # print(list(eval('all_df'+str(j))['variance'].values))
            bars = ax.bar(x=list(range(j-1, 5*len(eval('all_df'+str(j)).index), 5)),
                    height=list(eval('all_df'+str(j))['mean'].values),
                    width=1,
                    align="center",
                    label='等级: '+str(j),
                    )
            # 在每个柱形上方添加文字
            temp_hight = list(eval('all_df'+str(j))['variance'].values)
            temp_list = []
            for k, v in degre.items():
                temp_list.append(v)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height+temp_hight[i], temp_list[i][j-1], ha='center', va='bottom', color='red')

            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles, labels)
            plt.errorbar(list(range(j-1, 5*len(eval('all_df'+str(j)).index), 5)), list(eval('all_df'+str(j))['mean'].values), yerr=list(eval('all_df'+str(j))['variance'].values), fmt="o",
                         ecolor='black', color='black', elinewidth=1, capsize=5)
        col_dt=[]

        for i_ in col:
            i_ = re.sub(u"\\（.*?\\）|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", i_)
            col_dt.append(i_)
        plt.xticks(list(range(2, 5*len(all_df1.index), 5)), col_dt, rotation='vertical')
        plt.legend()
        plt.show()

        new_degre = {}
        # 首先将特定键值对添加到新字典
        for key in keys:
            if key in degre:
                new_degre[key] = degre[key]
        # 然后将原字典中的其他键值对添加到新字典
        for key, value in degre.items():
            if key not in keys:
                new_degre[key] = value
        print('-------------------------方差分析排序后-------------------------------')
        for key, value in new_degre.items():
            print(key, ':', value)
        print("最终筛选结果:{}".format(keys))
        return keys
    def uni(self, a, b, c):
        end1 = np.union1d(a, b)
        end2 = np.union1d(end1, c)
        print('最终选出成分为:')
        print(list(end2))
        return end2


if __name__ == '__main__':
    excel_file_x = './实验数据x.xlsx'  # 导入excel数据(x
    excel_file_y = './实验数据y.xlsx'  # 导入excel数据(y

    x_data = pd.read_excel(excel_file_x)  # 读取excel
    y_data = pd.read_excel(excel_file_y)  # 读取excel

    col = ['pH','有机质（g/kg）','全氮（mg/kg）','有效磷（mg/kg）','速效钾（mg/kg）','缓效钾（mg/kg）','有效锌（mg/kg）',
           '有效硼（mg/kg）','有效钼（mg/kg）','有效铜（mg/kg）','有效硅（mg/kg）','有效锰（mg/kg）','有效铁（mg/kg）']    ##x的列名（修改这里就行）

    label = ['可溶性固形物级别（数值越大品质越差）']  ##y的列名（修改这里就行）
                                                       ##因为之后的函数需要col和label这两个参数，所以提前单独写出来


    all_data = proce(x_data,col)  # 预处理

    data_x = all_data.loc[:, col]
    data_y = y_data.loc[:, label]

    p1 = Project1()
    cor_res = p1.cor(data_x, data_y)  # 相关分析
    end_res = p1.ent(data_x)  # 熵权法
    var_res = p1.var_analysis(data_x, data_y)  # 方差分析

    all_res = p1.uni(cor_res, end_res, var_res)  #  三个结果取并集