import pickle
import numpy as np
import pandas as pd

with open('data_NN.pkl', 'rb') as f:

    dataset = pickle.load(f)

if isinstance(dataset, dict):
    # 示例：把训练集特征存下来
    df = pd.DataFrame(dataset['x_train'].reshape(dataset['x_train'].shape[0], -1)) 
    
    # 3. 保存
    df.to_csv('my_dataset.csv', index=False)
    # 或者保存为 Excel (需要安装 openpyxl: pip install openpyxl)
    # df.to_excel('my_dataset.xlsx', index=False)
    print("文件已保存为 my_dataset.csv")