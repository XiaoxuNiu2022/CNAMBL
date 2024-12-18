import numpy as np
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, roc_auc_score,f1_score,precision_score,recall_score,confusion_matrix,r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,AdaBoostRegressor,RandomForestRegressor
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN,SMOTENC,SMOTEN
from imblearn.combine import SMOTEENN,SMOTETomek
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv1D, Bidirectional,Add, LSTM, Attention, Dense, Flatten, Dropout,Concatenate,Multiply
from tensorflow.keras.models import Sequential


from tensorflow.keras.layers import Input,Dense,Embedding,LSTM,GRU,BatchNormalization,Reshape,Conv1D,MaxPooling1D,Flatten,Input,Dropout,Multiply,Permute
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error as MAE


from tensorflow.keras import optimizers

# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import csv
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
tf.random.set_seed(2024)

def CN(X_train, y_train, X_test, y_test):
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    input_layer = Input(shape=(X_train.shape[1], 1))
    # CNN部分
    cnn_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)
    cnn_layer = Dense(32, activation='relu')(cnn_layer)

    cnn_layer = Dropout(0.2)(cnn_layer)
    output_layer = Dense(1)(cnn_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)

    # 5. 预测和评估
    y_pred  = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # 计算 RMSE

    return mae, mse, rmse

def AT(X_train, y_train, X_test, y_test):
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    input_layer = Input(shape=(X_train.shape[1], 1))

    attention_layer = Attention()([input_layer,input_layer])
    attention_layer = Flatten()(attention_layer)
    attention_layer = Dense(32,activation='relu')(attention_layer)

    concat_layer = Dropout(0.2)(attention_layer)
    output_layer = Dense(1)(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)
    # 5. 预测和评估
    y_pred  = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # 计算 RMSE
    return mae, mse, rmse

def BL(X_train, y_train, X_test, y_test):
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    input_layer = Input(shape=(X_train.shape[1], 1))

    bilstm_layer = Bidirectional(LSTM(32))(input_layer)
    bilstm_layer = Dense(32, activation='relu')(bilstm_layer)

    bilstm_layer= Dropout(0.2)(bilstm_layer)
    output_layer = Dense(1)(bilstm_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)
    # 5. 预测和评估
    y_pred  = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # 计算 RMSE
    return mae, mse, rmse

def CN_AT(X_train, y_train, X_test, y_test):
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    input_layer = Input(shape=(X_train.shape[1], 1))

    # CNN部分
    cnn_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)
    cnn_layer = Dense(32, activation='relu')(cnn_layer)


    attention_layer = Attention()([input_layer,input_layer])
    attention_layer = Flatten()(attention_layer)
    attention_layer = Dense(32,activation='relu')(attention_layer)

    # 特征
    concat_layer = Add()([cnn_layer, attention_layer])

    concat_layer =Dense(32, activation='relu')(concat_layer)
    concat_layer = Dropout(0.2)(concat_layer)
    output_layer = Dense(1)(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)
    # 5. 预测和评估
    y_pred  = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # 计算 RMSE
    return mae, mse, rmse

def AT_BL(X_train, y_train, X_test, y_test):
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    input_layer = Input(shape=(X_train.shape[1], 1))

    attention_layer = Attention()([input_layer,input_layer])
    attention_layer = Flatten()(attention_layer)
    attention_layer = Dense(32,activation='relu')(attention_layer)

    # # BiLSTM部分
    bilstm_layer = Bidirectional(LSTM(32))(input_layer)
    bilstm_layer = Dense(32, activation='relu')(bilstm_layer)

    # 融合attention和BiLSTM特征
    concat_layer = Add()([attention_layer,bilstm_layer])

    concat_layer =Dense(32, activation='relu')(concat_layer)
    concat_layer = Dropout(0.2)(concat_layer)
    output_layer = Dense(1)(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)
    # 5. 预测和评估
    y_pred  = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # 计算 RMSE
    return mae, mse, rmse

def CN_BL(X_train, y_train, X_test, y_test):
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    input_layer = Input(shape=(X_train.shape[1], 1))

    # CNN部分
    cnn_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)
    cnn_layer = Dense(32, activation='relu')(cnn_layer)

    # # BiLSTM部分
    bilstm_layer = Bidirectional(LSTM(32))(input_layer)
    bilstm_layer = Dense(32, activation='relu')(bilstm_layer)

    # 拼接CNN和BiLSTM特征
    concat_layer = Add()([cnn_layer, bilstm_layer])

    concat_layer =Dense(32, activation='relu')(concat_layer)
    concat_layer = Dropout(0.2)(concat_layer)
    output_layer = Dense(1)(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)
    # 5. 预测和评估
    y_pred  = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # 计算 RMSE

    # mae = round(mae, 3)
    # mse = round(mse, 3)
    # rmse = round(rmse, 3)
    return mae, mse, rmse


def CNAMBL(X_train, y_train, X_test, y_test):
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    input_layer = Input(shape=(X_train.shape[1], 1))
    # CNN部分
    cnn_layer = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = Flatten()(cnn_layer)
    cnn_layer = Dense(32, activation='relu')(cnn_layer)

    print(cnn_layer.shape)

    # attention部分
    attention_layer = Attention()([input_layer,input_layer])
    print(attention_layer.shape)
    attention_layer = Flatten()(attention_layer)
    print(attention_layer.shape)
    attention_layer = Dense(32,activation='relu')(attention_layer)
    print(attention_layer.shape)

    # # BiLSTM部分
    bilstm_layer = Bidirectional(LSTM(32))(input_layer)
    bilstm_layer = Dense(32, activation='relu')(bilstm_layer)
    print(bilstm_layer.shape)

    # 融合特征
    concat_layer = Add()([cnn_layer, attention_layer,bilstm_layer])#Concatenate()
    # concat_layer = Concatenate()([cnn_layer, attention_layer,bilstm_layer])#Concatenate()
    # concat_layer = Multiply()([cnn_layer, attention_layer,bilstm_layer])#Concatenate()

    concat_layer =Dense(32, activation='relu')(concat_layer)
    concat_layer = Dropout(0.2)(concat_layer)

    output_layer = Dense(1)(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error',metrics=['mae'])
    model.fit(X_train, y_train, epochs=100, batch_size=32,validation_split=0.2)

    # 5. 预测和评估
    y_pred  = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # 计算 RMSE

    # mae = round(mae, 3)
    # mse = round(mse, 3)
    # rmse = round(rmse, 3)

    return mae, mse, rmse

# 准备数据

data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_Java-Chassis1.3.1_2.0.2.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_jedis3.5.2_3.6.3.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_joda-time_2.1_2.2.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_lwjgl3.2.3_3.3.1.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_mina2.1.6_2.2.1.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_swagger-core1.6.2_2.1.4.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_swagger-core1.6.6_2.1.13.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_tomcat7.0.108_9.0.75.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_tomcat9.0.81_9.0.82.csv')
# data=pd.read_csv('../data_preprocessing/file_level_construct_lable_data/file_metrics_lable_Xchart3.5.4_3.6.1.csv')
# print(data)

X = data.iloc[:, 1:-1]
# 数据预处理
X=MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X)
# print(X)
y=data.iloc[:,-1:]

models_dict={
             "CN":CN,
             "AT":AT,
             "BL":BL,
             "CN_AT":CN_AT,
             "CN_BL":CN_BL,
             "AT_BL":AT_BL,
             "CNAMBL":CNAMBL,
            }
# 用于保存所有模型的平均值
all_models_avg = []
for model_name, model_func in models_dict.items():
    mae_scores = []
    mse_scores = []
    rmse_scores = []


    kf = KFold(n_splits=5, shuffle=True,random_state=2024)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        mae,mse,rmse=model_func(X_train,y_train,X_test,y_test)

        # r2_scores.append(r2)
        mae_scores.append(mae)
        mse_scores.append(mse)
        rmse_scores.append(rmse)


    # avg_mae = round(np.mean(mae_scores), 3)
    # avg_mse = round(np.mean(mse_scores), 3)
    # avg_rmse = round(np.mean(rmse_scores), 3)
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)

    all_models_avg.append([model_name,avg_mae,avg_mse,avg_rmse])
    print("model_name,avg_mae,avg_mse,avg_rmse")
    print(model_name,avg_mae,avg_mse,avg_rmse)


with open('all_models_avg_1016.csv', mode='w', newline='') as avg_file:
    writer = csv.writer(avg_file)
    writer.writerow(['model', 'avg_mae', 'avg_mse', 'avg_rmse'])  # 写入表头
    # writer.writerow(['model', 'avg_mae', 'avg_rmse'])  # 写入表头
    writer.writerows(all_models_avg)  # 写入所有模型的平均值





