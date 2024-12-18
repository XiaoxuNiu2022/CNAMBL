from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Attention, Bidirectional, LSTM, Add, Dropout
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
import csv
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(2024)

data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_Java-Chassis1.3.1_2.0.2.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_jedis3.5.2_3.6.3.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_joda-time_2.1_2.2.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_lwjgl3.2.3_3.3.1.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_mina2.1.6_2.2.1.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_swagger-core1.6.2_2.1.4.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_swagger-core1.6.6_2.1.13.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_tomcat7.0.108_9.0.75.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_tomcat9.0.81_9.0.82.csv')
# # data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_Xchart3.5.4_3.6.1.csv')
# # print(data)

X = data.iloc[:, 1:-1]
# 数据预处理
X=MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X)
# print(X)
y=data.iloc[:,-1:]

mae_scores = []
mse_scores = []


models_dict={
             "SVM":SVR(),
             "DT":DecisionTreeRegressor(random_state=2024),
             "RF":RandomForestRegressor(random_state=2024),
             "MLP":MLPRegressor(random_state=2024),
            }

# 用于保存所有模型的平均值
all_models_avg = []
kf = KFold(n_splits=5, shuffle=True,random_state=2024)
for model_name, model_func in models_dict.items():
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train Random Forest Regressor

        model_func.fit(X_train, y_train)

        # Make predictions
        y_pred = model_func.predict(X_test)

        # Evaluate the model
        mae= mean_absolute_error(y_test,y_pred)
        mse = mean_squared_error(y_test, y_pred)

        mae_scores.append(mae)
        mse_scores.append(mse)


    avg_mae = round(np.mean(mae_scores), 3)
    avg_mse = round(np.mean(mse_scores), 3)
    all_models_avg.append([model_name, avg_mae, avg_mse])
    print("model_name,avg_mae,avg_mse")
    print(model_name,avg_mae,avg_mse)
print(all_models_avg)


# with open('./ML_model/tomcat2_1016.csv', mode='w', newline='') as avg_file:
with open('./ML_model/tomcat2_1016.csv', mode='w', newline='') as avg_file:
    writer = csv.writer(avg_file)
    writer.writerow(['model', 'avg_mae', 'avg_mse'])  # 写入表头
    # writer.writerow(['model', 'avg_mae', 'avg_rmse'])  # 写入表头
    writer.writerows(all_models_avg)  # 写入所有模型的平均值
