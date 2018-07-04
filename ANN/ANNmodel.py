import os
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import scipy.stats
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
from sympy import *

# 定义模型所在路径
epochs = 3000
mm = MinMaxScaler([-1, 1])
ss = StandardScaler()



def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path)  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("{}: {}".format(key, value))

            print("    Dataset:")
            for name, d in g.items(): # 读取各层储存具体信息的Dataset类
                print("{}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
    finally:
        f.close()

def draw_data(data):
    plt.scatter([_ for _ in range(data.shape[0])], data)
    plt.show()

# input: 输入指标，为list形式
# weights：每层的权重，字典形式
# derivate_target: 求偏导目标的索引
# return DerivativeExpression
def DerivativeExpression(sympy_X, weights, Target):
    assert type(sympy_X) == list
    assert type(weights) == dict
    assert type(Target) == Symbol

    X = np.array(sympy_X)
    W1 = weights['sigmoid'][0]
    b1 = weights['sigmoid'][1]
    # print('sigmoid层的权重值为：')
    # print(W1)
    # print('sigmoid层的偏差为：')
    # print(b1)

    W2 = weights['linear'][0]
    b2 = weights['linear'][1]
    # print('linear层的权重值为：')
    # print(W2)
    # print('linear层的偏差为：')
    # print(b2)

    fa = np.dot(X, W1) + b1
    active_func = np.array([1 / (1 - exp(-_)) for _ in fa])
    fb = np.dot(active_func, W2) + b2
    # print("the derivation of ", input[derivate_target_index])
    # print(diff(fb, X[derivate_target_index]))
    return diff(fb[0], Target)

# Expression: the derivative Expression of Function
# Target: the name of derivative target, type is "sympy"
# InputIndex: the index data of derivative target, type is "list"
# InputIndex: the data of derivative target, type is "np.ndarray"
# return the calculate value
def DerivativeValue(Expression, Target, sympy_X, InputData):
    assert type(sympy_X) == list
    assert type(InputData) == np.ndarray
    assert len(sympy_X) == InputData.shape[1]
    assert type(Target) == Symbol


    DerValue = []
    for RawNo in range(InputData.shape[0]):
        TargetValue = list(InputData[RawNo])
        data = dict(zip(sympy_X, TargetValue))
        value = Expression.evalf(subs=data)
        print("raws value is :", value)
        DerValue.append(value)
    return DerValue



def ANN_Model(X, Y, test_X, test_y, input_dim):

    """----------测试集原数据作图----------------"""
    # plt.figure(0)  # 创建图表1
    # plt.title('observe')
    # plt.scatter([_ for _ in range(test_y.shape[0])], test_y)

    trainflag = 0

    if trainflag == 1:
        """----------配置网络模型----------------"""
        # 配置网络结构
        model = Sequential()

        # 第一隐藏层的配置：输入17，输出20
        model.add(Dense(64, input_dim=input_dim, activation='sigmoid'))
        # model.add(Dense(32, activation='sigmoid'))
        # SingleModel.add(Dense(20, activation='sigmoid'))
        model.add(Dense(1))

        # 编译模型，指明代价函数和更新方法
        sgd = optimizers.SGD(lr=0.5, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(loss='mae', optimizer=sgd, metrics=['mae'])

        """----------训练模型--------------------"""
        print("training starts.....")
        model.fit(X, Y, epochs=epochs, verbose=1, batch_size=64)

        """----------评估模型--------------------"""
        # 用测试集去评估模型的准确度
        cost = model.evaluate(test_X, test_y)
        print('\nTest accuracy:', cost)

        """----------模型存储--------------------"""
        save_model(model, weight_file_path)
        print('R^2:')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_y.T[0], model.predict(test_X).T[0])
        print(r_value**2)

    else:
        model = load_model(weight_file_path)

        # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_y.T[0], SingleModel.predict(test_X).T[0])
        # print("R^2:", r_value**2)
        """----------计算R^2--------------------"""
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(test_y.T[0], model.predict(test_X).T[0])
        print(r_value**2)
        """----------预测作图--------------------"""
        # plt.figure(1)  # 创建图表1
        # plt.title('predict')
        # plt.scatter([_ for _ in range(test_y.shape[0])], model.predict(test_X))
        # plt.show()

        """----------获取权重值--------------------"""
        calculate_derivation = 1
        if calculate_derivation == 1:
            weights = {}
            for layer in model.layers:
                weight = layer.get_weights()
                info = layer.get_config()
                # print(info['activation'])
                weights[info['activation']] = weight

            DerivativeData = {}
            sympy_X = [Symbol(_) for _ in InputIndex]
            # 需要求导的指标索引
            for index in range(len(InputIndex)):
                print('start to calculate: ', InputIndex[index])
                Expression = DerivativeExpression(sympy_X, weights, sympy_X[index])
                # print(type(Expression))
                # 从test数组中取出每一行数据
                Target = Symbol(InputIndex[index])
                DerivativeData[InputIndex[index]] = DerivativeValue(Expression, Target, sympy_X, test_X)
            # print(DerivativeData)
            DerivativeData['NEE'] = list(test_y.T[0])
            result = pd.DataFrame(DerivativeData)

            result.to_csv('data/'+data_path+'.csv')

        elif calculate_derivation == 0:
            dataShow = pd.read_csv('data/result.csv')
            draw_y = dataShow.pop('NEE').values.T
            draw_x = dataShow.values
            cloumns = dataShow.columns.tolist()

            for index in range(len(cloumns)):
                plt.figure(index)  # 创建图表1
                y = draw_y
                x = draw_x[:, index]
                plt.xlabel(cloumns[index])
                plt.ylabel("NEE-" + cloumns[index])
                plt.scatter(x, y)
                plt.savefig(picture_path + cloumns[index] + ".png")
        else:
            pass

def draw_pic(datafile):
    dataShow = pd.read_csv(datafile)
    draw_y = dataShow.pop('NEE').values.T
    draw_x = mm.fit_transform(dataShow.values)

    cloumns = dataShow.columns.tolist()

    for index in range(len(cloumns)):
        plt.figure(index)  # 创建图表1
        y = draw_y
        x = draw_x[:, index]
        plt.xlabel(cloumns[index])
        plt.ylabel("NEE-" + cloumns[index])
        plt.scatter(x, y)
        plt.savefig("res/" + cloumns[index] + ".png")

if __name__ == '__main__':


    # 屏蔽waring信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from data_preprocess import *
    """----------加载数据集-------------"""
    # InputIndex = [
    #     'PAR_dn_Avg', 'Ta_Avg', 'soil_T_1_10cm_Avg', 'ustar', 'VWC',
    #     'VPD'
    # ]
    OutputIndex = ['co2_flux']
    # data = pd.read_csv('data/growRes2012.csv')

    grow_data, ungrow_data, InputIndex = seperate_grow('data/growRes2012.csv')
    ungrow = 0
    # for ungrow in [0, 1]:
    if ungrow == 0:
        ungrow_data.dropna(inplace=True)
        weight_file_path = 'MultiplyModel/ANN_ungrow.h5'
        picture_path = 'ungrow/'
        data_path = 'ungrow'
        data = ungrow_data.copy()
    else:
        grow_data.dropna(inplace=True)
        weight_file_path = 'MultiplyModel/ANN_grow.h5'
        picture_path = 'grow/'
        data_path = 'grow'
        data = grow_data.copy()

    # 分割训练集合验证集，test_size=0.4代表从总的数据集合train中随机选取40%作为验证集，随机种子为0
    InputIndex = [
        'PAR_dn_Avg',
        'Ta_Avg',
        'soil_T_1_10cm_Avg',
        'ustar',
        'VWC',
        'VPD'
    ]
    train = ss.fit_transform(data[InputIndex])
    target = mm.fit_transform(data[OutputIndex])
    print(train.shape)
    trX, teX, trY, teY = train_test_split(train, target, test_size=0.5)
    # X = ss.fit_transform(trX)
    # Y = ss.fit_transform(trY)
    # test_X = mm.transform(teX)
    # test_y = mm.transform(teY)
    # ANN_Model(trX, trY, teX, teY, len(InputIndex))
    datafile = 'data/ungrow1.csv'
    draw_pic(datafile)