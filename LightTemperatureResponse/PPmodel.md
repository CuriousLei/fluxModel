- 安装包版本：
> numpy >= 1.13.1 <br> 
pandas >= 0.20.3 <br>
rpy2 >= 2.8.6 <br>
R version 3.3.2

- 初始化类输入数据:
> 
训练集白天数据：day_data <DataFrame> <br>
训练集晚上数据: night_data <DataFrame> <br>
测试集数据: test_data <DataFrame>

- PP model expression：
```math

CO2_{light} = {(a_0+a_1*VPD) * PAR * (p_0+p_1 * VPD) \over a_0*PAR + (p_0+p_2*VPD)}  + c

CO2_{temperature} = ((R0 + R1 * VPD) * e^{b*{T_a}})

```

- Trandition model expression：
```math

CO2_{light} = {a * PAR * b \over {a*PAR + b} } + c

CO2_{temperature} = ((R0 + R1 * VPD) * e^{b*{T_a}})

```

==在该模型中，首先选用PP模型拟合数据，PP模型拟合失败，采用传统模型继续拟合。<br>若要直接选用传统模型进行拟合，直接初始化传统模型类即可==