import csv
import matplotlib.pyplot as plt
import pandas as pd


df1 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_dense.csv')
df2 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse.csv')
df3 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/noher_dense.csv')
df4 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/noher_sparse.csv')

df5 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse_tau_0.1.csv')
df6 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse_tau_0.01.csv')
df7 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse_tau_0.0001.csv')
df8 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse_hard_update.csv')

df9 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse_OUnoise.csv')
df10 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse_HCnoise.csv')
df11 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/her_sparse_FGnoise.csv')


df20 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/DDPG_1.csv')
df21 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/TD3_2.csv')
df22 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/SAC_1.csv')


x=df1["Step"]
her_dense=df1['Value']
her_sparse=df2['Value']
noher_dense=df3['Value']
noher_sparse=df4['Value']

tau_01=df5['Value']
tau_001=df6['Value']
tau_0001=df2['Value']
tau_00001=df7['Value']
hard_decision=df8['Value']

normal_noise=df2['Value']
OUnoise=df9['Value']
HCnoise=df10['Value']
FGnoise=df11['Value']

DDPG=df20['Value']
TD3=df21['Value']
SAC=df22['Value']



#dense vs sparse no her
plt.figure(1)
plt.plot(x,noher_dense,label="Dense reward")
plt.plot(x,noher_sparse,label="Sparse reward")  #第一步：添加label参数
plt.xlabel("Episode")
plt.ylabel("Success rate")

#添加图例
plt.legend(loc=0) #第二部：添加图例(第一种显示中文的方式)
 #prop显示中文，loc确定图例出现在哪一个位置
#plt.legend(prop=my_font,loc=0)#第二种显示中文的方式


#dense vs sparse with her
plt.figure(2)
plt.plot(x,her_dense,label="Dense reward")
plt.plot(x,her_sparse,label="Sparse reward")  #第一步：添加label参数
plt.xlabel("Episode")
plt.ylabel("Success rate")
#plt.title("DDPG+HER success rate of different reward")
plt.legend(loc=0)


# spares 有无her
plt.figure(3)
plt.plot(x,noher_sparse,label="Without HER")
plt.plot(x,her_sparse,label="With HER")
plt.xlabel("Episode")
plt.ylabel("Success rate")
#plt.title("Adding HER for sparse reward")
plt.legend(loc=0)


# compare with sb3
plt.figure(4)
plt.plot(x[:500],DDPG[:500],label="Baseline3 DDPG")
plt.plot(x[:500],TD3[:500],label="Baseline3 TD3")
plt.plot(x[:500],SAC[:500],label="Baseline3 SAC")
plt.plot(x[:500],her_sparse[:500],label="This work")
plt.xlabel("Episode")
plt.ylabel("Success rate")
#plt.title("Compare with RF algorithm provided in SB3")
plt.legend(loc=0)

# 对比tau
plt.figure(5)
plt.plot(x,tau_01,label="tau=0.1")
plt.plot(x,tau_001,label="tau=0.01")
plt.plot(x,tau_0001,label="tau=0.001")
plt.plot(x,tau_00001,label="tau=0.0001")
plt.plot(x,hard_decision,label="Hard Update")
plt.xlabel("Episode")
plt.ylabel("Success rate")
#plt.title("Success rate of different update ratio tau")
plt.legend(loc=0)

# Noise
plt.figure(6)
plt.plot(x,normal_noise,label="Normal noise")
plt.plot(x,OUnoise,label="OU noise")
plt.plot(x,HCnoise,label="Hill-climbing noise")
plt.plot(x,FGnoise,label="Fine-grained noise")
plt.xlabel("Episode")
plt.ylabel("Success rate")
#plt.title("Success rate of different noise")
plt.legend(loc=0)




plt.show()






