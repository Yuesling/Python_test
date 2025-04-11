import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ======= 特征工程 =======
def compute_additional_features(numbers):
    """
    计算开奖号码的额外特征
    :param numbers: 原始开奖号码 [[9, 9, 7], [4, 2, 6], ...]
    :return: 包含和值、跨度、奇偶比例等的新特征
    """
    features = []
    for number_set in numbers:
        number_set = list(map(int, number_set))  # 确保是整数列表

        # 和值
        total_sum = sum(number_set)
        # 跨度
        span = max(number_set) - min(number_set)
        # 奇偶个数
        odd_count = len([num for num in number_set if num % 2 != 0])  # 奇数个数
        even_count = len(number_set) - odd_count  # 偶数个数
        # 最大值和最小值
        max_val = max(number_set)
        min_val = min(number_set)
        # 号码间距的平均值和标准差
        distances = [abs(number_set[i] - number_set[i + 1]) for i in range(len(number_set) - 1)]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        # 质数特征
        prime_count = len([num for num in number_set if num in [2, 3, 5, 7]])

        # 将上述所有特征添加到列表中
        features.append([
            total_sum, span, odd_count, even_count,
            max_val, min_val, mean_distance, std_distance,
            prime_count
        ])

    return np.array(features)


# ======= 数据处理 =======
def load_and_process_data(filename):
    """
    加载数据并计算相关特征
    :param filename: CSV 文件路径
    :return: 归一化的号码序列数据和对应额外特征
    """
    # 从 CSV 文件加载数据
    df = pd.read_csv(filename)

    # 获取开奖号码，并转为二维列表 [[9, 9, 7], [4, 2, 6], ...]
    df['开奖号码'] = df['开奖号码'].apply(eval)  # 转换为 [9, 9, 7] 格式
    numbers = np.array(df['开奖号码'].tolist())  # 转为二维数组

    # 计算额外特征
    additional_features = compute_additional_features(numbers)  # [[和值, 跨度, ...], ...]

    # 将开奖号码（原始特征）和额外特征拼接到一起
    combined_features = np.concatenate((numbers, additional_features), axis=1)

    # 数据归一化（整体归一化）
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_combined_features = scaler.fit_transform(combined_features)

    return scaled_combined_features, scaler


def create_sequences(data, look_back=10):
    """
    创建时间序列数据
    :param data: 归一化的数据
    :param look_back: 用于预测的时间窗口大小（最近期数）
    :return: 输入 (X) 和输出 (Y) 数据
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])  # 输入特征（历史 look_back 个时间步）
        Y.append(data[i + look_back, :3])  # 只预测开奖号码
    return np.array(X), np.array(Y)


# ======= 模型构建 =======
def build_lstm_model(input_shape, dropout_rate=0.2):
    """
    构建 LSTM 模型
    :param input_shape: 输入数据的形状 (时间步长, 特征数)
    :param dropout_rate: Dropout 比例，防止过拟合
    :return: 构建好的 LSTM 模型
    """
    model = Sequential([
        Input(shape=input_shape),  # 使用 Input 层定义输入形状
        LSTM(64, return_sequences=True),  # 第一层 LSTM
        Dropout(dropout_rate),  # 随机丢弃层，用来防止过拟合
        LSTM(64, return_sequences=False),  # 第二层 LSTM
        Dropout(dropout_rate),
        Dense(64, activation='relu'),  # 全连接层
        Dense(3, activation='linear')  # 输出层，预测 3 个开奖号码
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ======= 主程序 - 整体流程 =======
def main():

    # 加载并预处理数据
    filename = r"D:\Pythonprojet\lottery_results_all_pages.csv"  # 替换为你的文件路径
    data, scaler = load_and_process_data(filename)

    # 创建时间序列数据
    look_back = 10
    X, Y = create_sequences(data, look_back)

    # 划分训练集和验证集
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 构建 LSTM 模型
    model = build_lstm_model((X.shape[1], X.shape[2]), dropout_rate=0.2)
    print(model.summary())  # 打印模型结构

    # 使用早停策略
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型
    model.fit(
        X_train, Y_train,
        epochs=100, batch_size=32,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    # 使用模型预测
    last_data = data[-look_back:]
    last_data = np.expand_dims(last_data, axis=0)  # 增加 batch 维度
    prediction = model.predict(last_data)

    # 反归一化结果
    predicted_result = np.concatenate([prediction, np.zeros((1, data.shape[1] - 3))], axis=1)
    predicted_number = scaler.inverse_transform(predicted_result)
    predicted_number = np.round(predicted_number[0, :3]).astype(int)

    print("预测的下一期开奖号码：", predicted_number)


if __name__ == "__main__":
    main()
