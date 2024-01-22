import numpy as np
import matplotlib.pyplot as plt
from gmplot import gmplot

import numpy as np

def gm11(x0, predict_num):    
    n = len(x0)
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2

    y = x0[1:]
    x = z1

    k = ((n - 1) * np.sum(x * y) - np.sum(x) * np.sum(y)) / ((n - 1) * np.sum(x * x) - np.sum(x) * np.sum(x))
    b = (np.sum(x * x) * np.sum(y) - np.sum(x) * np.sum(x * y)) / ((n - 1) * np.sum(x * x) - np.sum(x) * np.sum(x))
    a = -k

    print('现在进行GM(1,1)预测的原始数据是: ')
    print(x0)
    print(f'最小二乘法拟合得到的发展系数为 {a}，灰作用量是 {b}')
    print('***************分割线***************')

    x0_hat = np.zeros(n)
    x0_hat[0] = x0[0]

    for m in range(n - 1):
        x0_hat[m + 1] = (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * (m+1))
        # print(f'第 {m + 1} 个预测值为 {x0_hat[m + 1]}')

    result = np.zeros(predict_num)

    for i in range(predict_num):
        result[i] = (1 - np.exp(a)) * (x0[0] - b / a) * np.exp(-a * (n + i))

    absolute_residuals = x0[1:] - x0_hat[1:]
    relative_residuals = np.abs(absolute_residuals) / x0[1:]

    class_ratio = x0[1:] / x0[:-1]
    eta = np.abs(1 - (1 - 0.5 * a) / (1 + 0.5 * a) * (1 / class_ratio))

    return result, x0_hat, relative_residuals, eta


# 构造时间序列数据
year = np.arange(1995, 2005)
data = np.array([174,179,183,189,207,234,220.5,256,270,285])

# plt.plot(year, data)
# plt.show()

error = 0
# 时间序列不存在负数
if np.any(data < 0):
    error = 1
    print('Error: Data must be non-negative.')
# 序列长度不能太短
if len(data)<=3:
    error = 1
    print('Error: Data must contain at least 3 points.')
# 序列不能太长
if len(data)>10:
    error = 1
    print('Error: Data must contain no more than 10 points.')

# 对一次累加数据进行准指数规律检验
if error==0:
    data_cum = np.cumsum(data)
    rho = data[1:]/data_cum[:-1]
    plt.plot(year[1:], rho)
    plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5')
    plt.title('rho')
    plt.show()
    
    # 计算rho小于0.5的占比
    filtered_data = rho[rho < 0.5]
    percentage1 = len(filtered_data) / len(rho) * 100
    print(f"数据小于0.5的占比: {percentage1}%")
    
    # 计算rho除去前两个时间点小于0.5的占比
    filtered_data = rho[2:][rho[2:] < 0.5]
    percentage2 = len(filtered_data) / len(rho[2:]) * 100
    print(f"除去前两个时间点，数据小于0.5的占比: {percentage2}%")
    
    if percentage1>60 and percentage2>90:
        print('数据符合准指数规律')
    else:
        print('数据不符合准指数规律!!!!!!!!!!!')
        error=1
    
if error==0:
    if len(data)>7:
        test_num = 3
    else:
        test_num = 2
    train_data = data[:-test_num]
    test_data = data[-test_num:]
    result, x0_hat, relative_residuals, eta = gm11(train_data, test_num)
    test_year = year[-test_num:]
    print(result)
    # 绘制原始数据
    plt.plot(test_year, test_data, label='Original Curve', color='blue', linestyle='-')
    # 绘制拟合数据
    plt.plot(test_year, result, label='Nihe Curve', color='red', linestyle='--')
    plt.show()
    
    #预测
    predict_num = int(input('请输入你要往后面预测的期数： '))
    result, x0_hat, relative_residuals, eta = gm11(data, predict_num)

    # 输出使用模型预测出来的结果
    print('------------------------------------------------------------')
    print('对原始数据的拟合结果：')
    for i in range(len(data)):
        print(f'{year[i]} ： {x0_hat[i]}')

    print(f'往后预测{predict_num}期的得到的结果：')
    for i in range(predict_num):
        print(f'{year[-1] + i + 1} ： {result[i]}')
        
# 绘制相对残差和级比偏差的图形
plt.figure(4, figsize=(8, 8))

# 子图1：相对残差
plt.subplot(2, 1, 1)
plt.plot(year[1:], relative_residuals, '*-')
plt.grid(True)
plt.legend(['relative_residuals'])
plt.xlabel('year')
plt.xticks(year[1:])
plt.title('relative residuals and jibi bias')

# 子图2：级比偏差
plt.subplot(2, 1, 2)
plt.plot(year[1:], eta, 'o-')
plt.grid(True)
plt.legend(['级比偏差'])
plt.xlabel('年份')
plt.xticks(year[1:])

# 输出对原数据拟合的评价结果
print('\n****下面将输出对原数据拟合的评价结果***\n')

# 残差检验
average_relative_residuals = np.mean(relative_residuals)
print(f'平均相对残差为{average_relative_residuals}')
if average_relative_residuals < 0.1:
    print('残差检验的结果表明：该模型对原数据的拟合程度非常不错')
elif average_relative_residuals < 0.2:
    print('残差检验的结果表明：该模型对原数据的拟合程度达到一般要求')
else:
    print('残差检验的结果表明：该模型对原数据的拟合程度不太好，建议使用其他模型预测')

# 级比偏差检验
average_eta = np.mean(eta)
print(f'平均级比偏差为{average_eta}')
if average_eta < 0.1:
    print('级比偏差检验的结果表明：该模型对原数据的拟合程度非常不错')
elif average_eta < 0.2:
    print('级比偏差检验的结果表明：该模型对原数据的拟合程度达到一般要求')
else:
    print('级比偏差检验的结果表明：该模型对原数据的拟合程度不太好，建议使用其他模型预测')
print('\n------------------------------------------------------------\n')

# 绘制最终的预测效果图
plt.figure(5, figsize=(8, 4))
plt.plot(year, data, '-o', label='原始数据')
plt.plot(year, x0_hat, '-*m', label='拟合数据')
plt.plot(np.arange(year[-1] + 1, year[-1] + 1 + predict_num), result, '-*b', label='预测数据')
plt.plot([year[-1], year[-1] + 1], [data[-1], result[0]], '-*b')
plt.grid(True)
plt.legend(['原始数据', '拟合数据', '预测数据'])
plt.xticks(np.arange(year[0], year[-1] + predict_num + 1))
plt.xlabel('年份')
plt.ylabel('排污总量')
plt.title('最终的预测效果图')
plt.show()