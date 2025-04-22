import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def load_and_process_data(filename):
    # 加载 CSV 文件
    df = pd.read_csv(filename)
    # 解析 '开奖号码' 列为列表，并将每个数字转换为整数
    def parse_draw_number(x):
        try:
            # 将字符串转换为列表，并确保每个元素是整数
            return [int(item) for item in literal_eval(x)]
        except (ValueError, SyntaxError) as e:
            # 如果解析失败，返回 None 或您认为合适的值
            print(f"解析失败: {x}")
            return None
    # 解析 '开奖号码' 列
    df['开奖号码'] = df['开奖号码'].apply(parse_draw_number)
    # 检查是否有无法解析的数据
    if df['开奖号码'].isnull().any():
        print("以下行的开奖号码无法解析为有效数字列表：")
        print(df[df['开奖号码'].isnull()])
        raise ValueError("部分 '开奖号码' 非有效数字列表，请检查数据！")
    # 增加手工特征（和值、奇数个数、偶数个数）
    df['和值'] = df['开奖号码'].apply(sum)
    df['奇数个数'] = df['开奖号码'].apply(lambda x: sum(1 for n in x if n % 2 != 0))
    df['偶数个数'] = df['开奖号码'].apply(lambda x: sum(1 for n in x if n % 2 == 0))

    # 将所有数据拼接成多维数组
    feature_data = np.array(df['开奖号码'].tolist())  # 转为数组
    additional_features = df[['和值', '奇数个数', '偶数个数']].values
    combined_data = np.hstack([feature_data, additional_features])
    # 转换为浮点数（便于归一化处理）
    combined_data = combined_data.astype(float)
    # 数据归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)
    return scaled_data, scaler



def create_sequences(data, look_back=30, num_classes=10):
    """
    创建时间序列输入和多分类输出。
    """
    X, Y = [], []

    for i in range(len(data) - look_back - 1):
        # 构造输入序列
        sequence = data[i: i + look_back]
        X.append(sequence)

        # 构造目标输出
        next_numbers = data[i + look_back, :3].astype(int)  # 假设前三列是开奖号码
        one_hot_encoded = np.zeros((len(next_numbers), num_classes))
        for index, num in enumerate(next_numbers):
            if num < num_classes:  # 确保号码在范围内
                one_hot_encoded[index][num] = 1
        Y.append(one_hot_encoded)

    return np.array(X), np.array(Y)


def build_lstm_model(input_shape, num_classes):
    """
    构建一个基于序列输入的改进型 LSTM 模型
    """
    inputs = Input(shape=input_shape)

    # 使用双向 LSTM 和正则化
    x = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001)))(inputs)
    x = Dropout(0.3)(x)  # 防止过拟合

    x = LSTM(256, return_sequences=False, kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)

    # 将输出转化为与目标值时间步相匹配的形状
    x = RepeatVector(3)(x)  # 修改为 3
    x = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001))(x)
    outputs = TimeDistributed(Dense(num_classes, activation="softmax"))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model




def apply_temperature(probs, temperature=1.0):
    probs = np.log(probs + 1e-8) / temperature  # 调整 logits
    exp_probs = np.exp(probs)
    return exp_probs / np.sum(exp_probs, axis=-1, keepdims=True)


def main():
    # 文件路径（替换为实际数据文件路径）
    filename = r"D:\Pythonprojet\lottery_results_all_pages.csv"

    # 加载并处理数据
    print("加载数据...")
    data, scaler = load_and_process_data(filename)

    # 参数定义
    look_back = 30  # 时间序列窗口长度
    num_classes = 10  # 数字范围是 0-9

    # 创建训练数据和目标结果
    print("创建时间序列...")
    X, Y = create_sequences(data, look_back, num_classes)

    print(f"数据形状：X_train={X.shape}, Y_train={Y.shape}")

    # 数据集拆分
    print("拆分数据集...")
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # 构建模型
    print("构建模型...")
    model = build_lstm_model((X.shape[1], X.shape[2]), num_classes)
    print(model.summary())

    # 配置回调函数
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    # 开始训练
    print("训练模型...")
    model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, Y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # 在测试集上评估模型
    print("评估在测试集上的性能...")
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print(f"测试集损失：{test_loss:.4f}, 测试集准确率：{test_accuracy:.4f}")

    # 用最后的时间序列预测未来一期数据
    print("预测未来开奖号码...")
    last_data = X[-1:]  # 提取最后一段时间序列
    prediction_probs = model.predict(last_data)

    # 使用温度缩放并解码
    temperature = 0.8  # 调节温度值
    decoded_numbers = []
    for i in range(prediction_probs.shape[1]):  # 针对每个位置的号码
        probs = apply_temperature(prediction_probs[0, i], temperature)
        predicted_number = np.random.choice(range(num_classes), p=probs[0])
        decoded_numbers.append(predicted_number)

    print("预测的下一期号码：", decoded_numbers)


if __name__ == "__main__":
    main()
