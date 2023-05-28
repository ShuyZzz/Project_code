import csv
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/X_GN.csv')
df2 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/Y_GN.csv')
df3 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/Z_GN.csv')
df4 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/V_X_GN.csv')
df5 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/V_Y_GN.csv')
df6 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/V_Z_GN.csv')

df11 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/X_OUN.csv')
df12 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/Y_OUN.csv')
df13 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/Z_OUN.csv')
df14 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/V_X_OUN.csv')
df15 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/V_Y_OUN.csv')
df16 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/V_Z_OUN.csv')

df21 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/Goal_X.csv')
df22 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/Goal_y.csv')
df23 = pd.read_csv('/Users/hll/PycharmProjects/shuyi/Thesis_project_shuyi/excel_data/velocity/Goal_Z.csv')

t=df1['Step']

v_x=df4['Value']
v_y=df5['Value']
v_z=df6['Value']

v_x_ou=df14['Value']
v_y_ou=df15['Value']
v_z_ou=df16['Value']

x=df1['Value']
y=df2['Value']
z=df3['Value']

x_ou=df11['Value']
y_ou=df12['Value']
z_ou=df13['Value']

t_x=df21['Value']
t_y=df22['Value']
t_z=df23['Value']



plt.figure(1)
#plot 1:
plt.subplot(3, 1, 1)
plt.plot(t,v_x)
plt.ylabel("Velocity_x")
#plot 2:
plt.subplot(3, 1, 2)
plt.plot(t,v_y)
plt.ylabel("Velocity_y")
#plot 3:
plt.subplot(3, 1, 3)
plt.plot(t,v_z)
plt.ylabel("Velocity_z")
plt.xlabel("Steps")



plt.figure(2)
#plot 1:
plt.subplot(3, 1, 1)
plt.plot(t,v_x_ou)
plt.ylabel("Velocity_x")
#plot 2:
plt.subplot(3, 1, 2)
plt.plot(t,v_y_ou)
plt.ylabel("Velocity_y")
#plot 3:
plt.subplot(3, 1, 3)
plt.plot(t,v_z_ou)
plt.ylabel("Velocity_z")
plt.xlabel("Steps")




plt.figure(3)
#plot 1:
plt.subplot(3, 1, 1)
plt.plot(t,v_x,label='nomal noise')
plt.plot(t,v_x_ou,label='OU noise')
plt.ylabel("Velocity_x")
plt.legend(loc=0)
#plot 2:
plt.subplot(3, 1, 2)
plt.plot(t,v_y)
plt.plot(t,v_y_ou)
plt.ylabel("Velocity_y")
plt.legend(loc=0)
#plot 3:
plt.subplot(3, 1, 3)
plt.plot(t,v_z)
plt.plot(t,v_z_ou)
plt.xlabel("Steps")
plt.ylabel("Velocity_z")
plt.legend(loc=0)


plt.figure(4)
#plot 1:
plt.subplot(3, 1, 1)
plt.plot(t,x_ou,label='End-effector')
plt.plot(t,t_x,label='Target')
plt.ylabel("Position_x")
plt.legend(loc=0)
#plot 2:
plt.subplot(3, 1, 2)
plt.plot(t,y_ou)
plt.plot(t,t_y)
plt.ylabel("Position_y")
#plot 3:
plt.subplot(3, 1, 3)
plt.plot(t,z_ou)
plt.plot(t,t_z)
plt.ylabel("Position_z")
plt.xlabel("Steps")


plt.figure(5)
plt.ylabel('Position')
#plot 1:
plt.subplot(3, 1, 1)
plt.plot(t,x,label='End-effector')
plt.plot(t,t_x,label='Target')
plt.ylabel("Position_x")
plt.legend(loc=0)
#plot 2:
plt.subplot(3, 1, 2)
plt.plot(t,y)
plt.plot(t,t_y)
plt.ylabel("Position_y")
#plot 3:
plt.subplot(3, 1, 3)
plt.plot(t,z)
plt.plot(t,t_z)
plt.ylabel("Position_z")
plt.xlabel("Steps")

plt.show()



