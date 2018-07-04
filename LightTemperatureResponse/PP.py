import numpy as np
import pandas as pd
from math import ceil
import random

import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
base = importr('base')
stats = importr('stats')

class NLS_PP:
    def __init__(self, day_data, night_data, test_data):
        self.day_data = day_data
        self.night_data = night_data
        self.test_data = test_data
        # 白天的数据
        self.test_day = test_data[test_data['PAR_dn_Avg'] >= 10]
        # 晚上的数据
        self.test_night = test_data[test_data['PAR_dn_Avg'] < 10]


    def PPLight(self, data, p):
        a0, a1, p0, p1, p2, c = p
        return (a0+a1*data['VPD'])*data['PAR_dn_Avg']*(p0+p1*data['VPD'])/(a0*data['PAR_dn_Avg']+(p0+p2*data['VPD'])) + c
    def PPLightR(self):
        return 'co2_flux ~ ((a0 + a1 * VPD) * PAR_dn_Avg * (p0+p1 * VPD)/(a0  * PAR_dn_Avg  + (p0+p2 * VPD)) + c)'


    def PPTemperature(self, data, p):
        R0, R1, b = p
        return (R0 + R1 * data['VPD']) * np.exp(b * data['Ta_Avg'])
    def PPTemperatureR(self):
        return 'co2_flux ~ ((R0 + R1 * VPD) * exp(b*Ta_Avg))'


    def PP_Lr_function(self, temp_data):
        mylist = r('list(a0=-0.04, a1=-0.001, p0=5, p1=-0.01, p2=-0.01, c=2)')
        r_dataframe = pandas2ri.py2ri(temp_data)
        A = stats.nls(
            self.PPLightR(),
            start=mylist,
            data=r_dataframe
        )
        a0, a1, p0, p1, p2, c = [None, None, None, None, None, None]

        # print(base.summary(A).rx2('coefficients'))
        pa0 = base.summary(A).rx2('coefficients')[18]
        pa1 = base.summary(A).rx2('coefficients')[19]
        pp0 = base.summary(A).rx2('coefficients')[20]
        pp1 = base.summary(A).rx2('coefficients')[21]
        pp2 = base.summary(A).rx2('coefficients')[22]
        pc = base.summary(A).rx2('coefficients')[23]
        if ((pa0 < 0.05) and (pa1 < 0.05) and (pp0 < 0.05) and (pp1 < 0.05) and (pp2 < 0.05) and (pc < 0.05)):
            a0 = base.summary(A).rx2('coefficients')[0]
            a1 = base.summary(A).rx2('coefficients')[1]
            p0 = base.summary(A).rx2('coefficients')[2]
            p1 = base.summary(A).rx2('coefficients')[3]
            p2 = base.summary(A).rx2('coefficients')[4]
            c = base.summary(A).rx2('coefficients')[5]
            return 'success', a0, a1, p0, p1, p2, c
        else:
            return 'failed', a0, a1, p0, p1, p2, c


    def PP_LR_model(self):
        temp_data = self.day_data[['co2_flux', 'PAR_dn_Avg', 'VPD']].copy()
        gap_data = self.test_day
        try:
            status,  a0, a1, p0, p1, p2, c = self.PP_Lr_function(temp_data)
            if status == 'success':
                print('PP光响应模型拟合成功')
                nls_light_data = self.PPLight(gap_data, [a0, a1, p0, p1, p2, c])
                return nls_light_data
            else:
                try:
                    print('2 如果PP的光响应模型没有拟合的话，那就用传统的光模型')
                    Trandition = NLS_Tr(self.day_data, self.night_data, self.test_data)

                    status, a, b, c = Trandition.LR_function(temp_data)
                    if status == 'success':
                        print('传统光响应模型拟合成功')
                        return Trandition.TrLight(gap_data, [a, b, c])
                    else:
                        print('PP光相应模型中每个参数的概率均未小于0.05，那就用PP的呼吸模型')
                        return self.PP_TR_model('day')
                except Exception as err:
                    # print(err)
                    print('如果传统光模型拟合失败，那就用PP的呼吸模型')
                    return self.PP_TR_model('day')
        except:
            try:
                print('2 如果PP的光响应模型拟合失败，那就用传统的光模型')
                Trandition = NLS_Tr(self.day_data, self.night_data, self.test_data)

                status, a, b, c = Trandition.LR_function(temp_data)
                if status == 'success':
                    print('传统光响应模型拟合成功')
                    return Trandition.TrLight(gap_data, [a, b, c])
                else:
                    print('传统光响应模型每个参数的概率均未小于0.05，那就用PP的呼吸模型')
                    return self.PP_TR_model('day')
            except Exception as err:
                # print(err)
                print('如果传统光模型拟合失败，那就用PP的呼吸模型')
                return self.PP_TR_model('day')


    def PP_Tr_function(self, temp_data):
        mylist = r('list(R0=5, R1=0.01, b=-0.04)')
        r_dataframe = pandas2ri.py2ri(temp_data)
        A = stats.nls(
            self.PPTemperatureR(),
            start=mylist,
            data=r_dataframe
        )

        R0 = base.summary(A).rx2('coefficients')[0]
        R1 = base.summary(A).rx2('coefficients')[1]
        b = base.summary(A).rx2('coefficients')[2]

        return R0, R1, b
    def PP_TR_model(self, flag):
        if flag == 'day':
            temp_data = self.day_data[['co2_flux', 'Ta_Avg', 'VPD']].copy()
            gap_data = self.test_day
        else:
            temp_data = self.night_data[['co2_flux', 'Ta_Avg', 'VPD']].copy()
            gap_data = self.test_night

        try:
            R0, R1, b = self.PP_Tr_function(temp_data)

            print('PP呼吸模型拟合成功')
            return self.PPTemperature(gap_data, [R0, R1, b])
        except Exception as err:
            # print(err)
            print('如果PP的呼吸响应模型拟合失败，那就用传统的呼吸模型')
            try:
                Trandition = NLS_Tr(self.day_data, self.night_data, self.test_data)
                a, b = Trandition.TR_function(temp_data)
                print('传统呼吸模型拟合成功')
                return Trandition.TrTemperature(gap_data, [a, b])
            except Exception as err:
                # print(err)
                print('如果传统呼吸模型也拟合失败，老子GG')

class NLS_Tr:
    def __init__(self, day_data, night_data, test_data):
        self.day_data = day_data
        self.night_data = night_data
        # 白天的数据
        self.test_day = test_data[test_data['PAR_dn_Avg'] >= 10]
        # 晚上的数据
        self.test_night = test_data[test_data['PAR_dn_Avg'] < 10]

    def TrLight(self, data, p):
        a, b, c = p
        return a*data['PAR_dn_Avg']*b/(a*data['PAR_dn_Avg']+b) + c
    def TrLightR(self):
        return 'co2_flux ~ (a * PAR_dn_Avg * b/(a  * PAR_dn_Avg + b) + c)'

    def TrTemperature(self, data, p, value='Ta_Avg'):
        a, b = p
        return a * np.exp(b * data[value])

    def TrTemperatureR(self, value='Ta_Avg'):
        return 'co2_flux ~ (a * exp(b*'+value+'))'


    def LR_function(self, temp_data):
        r_dataframe = pandas2ri.py2ri(temp_data)
        mylist = r('list(a=-0.04, b=-10, c=2)')
        a, b, c = [None, None, None]
        A = stats.nls(
            self.TrLightR(),
            start=mylist,
            data=r_dataframe
        )
        pa = base.summary(A).rx2('coefficients')[9]
        pb = base.summary(A).rx2('coefficients')[10]
        pc = base.summary(A).rx2('coefficients')[11]
        if (pa < 0.05) and (pb < 0.05) and (pc < 0.05):
            a = base.summary(A).rx2('coefficients')[0]
            b = base.summary(A).rx2('coefficients')[1]
            c = base.summary(A).rx2('coefficients')[2]
            return 'success', a, b, c
        else:
            return 'failed', a, b, c

    def Tr_LR_model(self):
        temp_data = self.day_data[['co2_flux', 'PAR_dn_Avg']].copy()
        gap_data = self.test_day
        try:
            status, a, b, c = self.LR_function(temp_data)
            if status == 'success':
                print('传统光响应方程拟合成功')
                return self.TrLight(gap_data, [a, b, c])
            else:
                print('传统光响应方程拟合失败')
                return self.Tr_TR_model('day')
        except Exception as err:
            return self.Tr_TR_model('day')


    def TR_function(self, temp_data, value='Ta_Avg'):
        mylist = r('list(a=10, b=-0.02)')
        r_dataframe = pandas2ri.py2ri(temp_data)
        A = stats.nls(
            self.TrTemperatureR(value),
            start=mylist,
            data=r_dataframe
        )

        a = base.summary(A).rx2('coefficients')[0]
        b = base.summary(A).rx2('coefficients')[1]
        return a, b
    def Tr_TR_model(self, flag):
        temp_data = self.night_data[['co2_flux', 'Ta_Avg']].copy()
        if flag == 'day':
            gap_data = self.test_day
        else:
            gap_data = self.test_night
        try:
            a, b = self.TR_function(temp_data)
            print('传统(空气)温度响应方程拟合成功')
            return self.TrTemperature(gap_data, [a, b])
        except Exception as err:
            temp_data = self.night_data[['co2_flux', 'soil_T_1_10cm_Avg']].copy()
            if flag == 'day':
                gap_data = self.test_day
            else:
                gap_data = self.test_night
            try:
                a, b = self.TR_function(temp_data, value='soil_T_1_10cm_Avg')
                print('传统(土壤表层)温度响应方程拟合成功')
                return self.TrTemperature(gap_data, [a, b], value='soil_T_1_10cm_Avg')
            except Exception as err:
                # 如果还不行，那就循环删减数据，直到拟合为止
                for number in range(temp_data.shape[0], -1, -2):
                    try:
                        temp_data = temp_data.tail(number)
                        a, b = self.TR_function(temp_data, value='soil_T_1_10cm_Avg')
                        print('传统(土壤表层)温度响应方程拟合成功')
                        return self.TrTemperature(gap_data, [a, b], value='soil_T_1_10cm_Avg')
                    except Exception as err:
                        continue