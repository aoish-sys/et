import matplotlib.pyplot as plt
import numpy as np

# 定义每个 rho 对应的数据
epochs = np.arange(1, 21)  # 20个周期，对应20个数据点

# 数据
rho_1_0 = [20.63, 28.16, 30.01, 31.66, 32.79, 33.01, 33.04, 32.73, 32.80, 33.08, 33.11, 33.21, 33.20, 33.31, 33.41, 33.23, 33.04, 33.53, 33.56, 33.55]
rho_0_8 = [19.99, 27.52, 30.63, 31.82, 32.41, 32.51, 33.01, 32.56, 32.60, 32.46, 32.54, 32.50, 32.48, 32.31, 32.41, 32.23, 32.04, 31.96, 32.23, 32.18]
rho_0_6 = [28.26, 29.97, 31.00, 31.57, 31.74, 31.83, 31.87, 31.86, 31.90, 31.95, 31.94, 31.91, 31.90, 31.83, 31.75, 31.80, 31.99, 31.86, 31.85, 31.85]
rho_0_4 = [19.99, 27.52, 30.63, 31.82, 32.41, 32.51, 33.01, 32.56, 32.60, 32.46, 32.54, 32.50, 32.48, 32.31, 32.41, 32.23, 32.04, 31.96, 32.23, 32.18]
rho_0_2 = [15.75, 16.03, 16.49, 16.56, 16.85, 17.03, 16.77, 16.82, 16.92, 17.04, 17.16, 16.91, 17.10, 17.16, 16.94, 16.99, 16.72, 16.80, 16.92, 17.18]
# 绘制图形
plt.figure(figsize=(12, 8))

plt.plot(epochs, rho_1_0, label='rho = 1.0', marker='o', color='blue')
plt.plot(epochs, rho_0_8, label='rho = 0.8', marker='o', color='green')
plt.plot(epochs, rho_0_6, label='rho = 0.6', marker='o', color='red')
plt.plot(epochs, rho_0_4, label='rho = 0.4', marker='o', color='purple')
plt.plot(epochs, rho_0_2, label='rho = 0.2', marker='o', color='black')
plt.plot(epochs, full_training, label='full_training', marker='o', color='pink')

plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('iid-Performance Comparison for Different Rho Values')
plt.xticks(ticks=epochs)  # 设置横坐标刻度为整数
plt.legend()
plt.grid(True)

# 保存图片到文件
plt.savefig('performance_comparison-iid.png')

# 显示图形
plt.show()
