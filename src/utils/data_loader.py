import pickle
import numpy as np
import pandas as pd

# 1. 打开并读取 pkl 文件
with open('../../data/data_NN.pkl', 'rb') as f:
    dataset = pickle.load(f)

if isinstance(dataset, dict):
    # 2. 打印看看里面到底有哪些“钥匙”（键名）
    print("🔍 这个数据集里包含的全部内容有：", dataset.keys())

    # --- 下面是原来的保存特征 X 的代码 ---
    df_x = pd.DataFrame(dataset['x_train'].reshape(dataset['x_train'].shape[0], -1)) 
    df_x.to_csv('my_dataset.csv', index=False)
    print("✅ 特征文件已保存为 my_dataset.csv")

    # --- 新增：寻找并保存标签 Y 的代码 ---
    # 假设标签的名字叫 'y_train'（如果不叫这个，你可以根据上面打印出的 keys 名字来改）
    if 'y_train' in dataset:
        y_data = dataset['y_train']
        print(f"🎯 找到了标签数据，它的形状是：{y_data.shape}")
        
        # 把标签存成另一个 CSV 文件
        df_y = pd.DataFrame(y_data)
        df_y.to_csv('my_labels.csv', index=False)
        print("✅ 标签文件已成功保存为 my_labels.csv 啦！")
    else:
        print("⚠️ 没有找到名为 'y_train' 的数据，请你看看第一行打印的 keys() 里哪个像标签数据？")