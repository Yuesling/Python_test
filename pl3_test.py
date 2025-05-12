import os
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def load_and_process_data(filename):
    """
    加载并处理彩票数据，包括解析开奖号码和添加额外特征的预处理。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件路径 {filename} 不存在，请检查后重试。")

    # 加载 CSV 文件
    df = pd.read_csv(filename)
    print("数据集加载完成。")

    def parse_draw_number(x):
        try:
            parsed = literal_eval(x)  # 将字符串转为 list
            if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
                raise ValueError(f"无效的数据格式：{x}")
            return [int(item) for item in parsed]
        except (ValueError, SyntaxError) as e:
            print(f"解析失败: {x}，错误: {e}")
            return None

    df['开奖号码'] = df['开奖号码'].apply(parse_draw_number)
    if df['开奖号码'].isnull().any():
        print("以下行的开奖号码无法解析为有效数字列表：")
        print(df[df['开奖号码'].isnull()])
        raise ValueError("部分 '开奖号码' 无法解析，请检查数据！")

    df['和值'] = df['开奖号码'].apply(sum)
    df['奇数个数'] = df['开奖号码'].apply(lambda x: sum(1 for n in x if n % 2 != 0))
    df['偶数个数'] = df['开奖号码'].apply(lambda x: sum(1 for n in x if n % 2 == 0))

    feature_data = np.array(df['开奖号码'].tolist())
    additional_features = df[['和值', '奇数个数', '偶数个数']].values
    combined_data = np.hstack([feature_data, additional_features]).astype(float)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)
    print(f"数据处理完成，数据形状：{scaled_data.shape}")
    return scaled_data, scaler


def create_sequences(data, look_back=30, num_classes=10):
    """
    构建时间序列输入（X）和多分类输出（Y）。
    """
    if len(data) < look_back + 1:
        raise ValueError(f"数据量太少，无法生成序列（要求至少 {look_back + 1} 条数据）")

    X, Y = [], []

    for i in range(len(data) - look_back - 1):
        X.append(data[i: i + look_back])
        next_numbers = data[i + look_back, :3].astype(int)

        one_hot_encoded = np.zeros((3, num_classes))
        for idx, num in enumerate(next_numbers):
            one_hot_encoded[idx, num] = 1 if 0 <= num < num_classes else 0

        Y.append(one_hot_encoded)

    X, Y = np.array(X), np.array(Y)
    print(f"序列生成完成，X 形状 = {X.shape}, Y 形状 = {Y.shape}")
    return X, Y


def compute_sample_weights(Y_train):
    # 获取实际存在的类别
    flat_labels = np.argmax(Y_train, axis=-1).flatten()
    unique_classes = np.unique(flat_labels)

    # 计算类别权重
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=flat_labels
    )

    # 创建样本权重（保持与Y_train相同的形状）
    sample_weights = np.ones_like(Y_train[..., 0])  # (samples, timesteps)

    # 应用权重
    weight_dict = dict(zip(unique_classes, class_weights))
    for t in range(Y_train.shape[1]):  # 遍历时间步
        labels_at_t = np.argmax(Y_train[:, t, :], axis=-1)
        for cls, weight in weight_dict.items():
            sample_weights[:, t][labels_at_t == cls] = weight

    print(f"样本权重形状: {sample_weights.shape}")  # 应为(样本数, 3)
    return sample_weights


def build_lstm_model(input_shape, num_classes):
    """
    构造改进型 Bidirectional LSTM 模型。
    """
    inputs = Input(shape=input_shape)

    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(inputs)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001))(x)

    x = RepeatVector(3)(x)
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))(x)
    outputs = TimeDistributed(Dense(num_classes, activation="softmax"))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    filename = r"D:\Pythonprojet\lottery_results_all_pages.csv"

    print("加载数据...")
    data, scaler = load_and_process_data(filename)

    look_back = 30
    num_classes = 10

    print("创建时间序列...")
    X, Y = create_sequences(data, look_back, num_classes)

    print("拆分数据集...")
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    print("计算样本权重...")
    sample_weights = compute_sample_weights(Y_train)

    print("构建模型...")
    model = build_lstm_model((X.shape[1], X.shape[2]), num_classes)
    print(model.summary())

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    print("开始训练模型...")
    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, Y_val),
        sample_weight=sample_weights,  # 现在是一维权重
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    print("评估模型性能...")
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print(f"测试集损失：{test_loss:.4f}, 测试集准确率：{test_accuracy:.4f}")

    print("预测未来开奖号码...")
    last_data = X[-1:]
    prediction_probs = model.predict(last_data)

    print("预测结果（未解码）：", prediction_probs)

    decoded_numbers = np.argmax(prediction_probs[0], axis=-1)
    print("预测的下一期号码：", decoded_numbers)


if __name__ == "__main__":
    main()
