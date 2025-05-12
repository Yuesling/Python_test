import numpy as np
import pandas as pd
import ast
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chisquare
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, LSTM, Conv1D,
                                     GlobalMaxPooling1D, Input,
                                     TimeDistributed, Reshape,
                                     concatenate, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        Callback)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ==================== 1. 数据加载与验证 ====================
def load_and_validate_data(filename):
    """加载并验证数据"""
    df = pd.read_csv(filename)
    print(f"数据量: {len(df)}行 | 列: {df.columns.tolist()}")

    # 解析开奖号码
    numbers = np.array([ast.literal_eval(x) for x in df['开奖号码']]).astype(np.int32)

    # 随机性检验
    flat_numbers = numbers.flatten()
    freq = Counter(flat_numbers)
    chi2, p = chisquare(list(freq.values()))
    print(f"随机性检验 p值={p:.4f} {'(随机)' if p > 0.05 else '(非随机)'}")

    # 可视化分布
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.hist(flat_numbers, bins=10, alpha=0.7)
    plt.title("号码分布")

    plt.subplot(122)
    plt.plot(numbers[:100])
    plt.title("近期走势")
    plt.tight_layout()
    plt.show()

    return numbers


# ==================== 2. 增强特征工程 ====================
def create_enhanced_features(numbers, window=20):
    """创建增强特征"""
    features = []
    for pos in range(3):
        df_pos = pd.DataFrame({'num': numbers[:, pos]})

        # 基础统计特征
        for w in [5, 10, 15, 20]:
            df_pos[f'mean_{w}'] = df_pos['num'].rolling(w).mean()
            df_pos[f'median_{w}'] = df_pos['num'].rolling(w).median()
            df_pos[f'skew_{w}'] = df_pos['num'].rolling(w).skew()
            df_pos[f'q75_{w}'] = df_pos['num'].rolling(w).quantile(0.75)
            df_pos[f'q25_{w}'] = df_pos['num'].rolling(w).quantile(0.25)

        # 趋势特征
        df_pos['trend_5'] = df_pos['num'].rolling(5).apply(
            lambda x: np.polyfit(range(5), x, 1)[0])
        df_pos['trend_10'] = df_pos['num'].rolling(10).apply(
            lambda x: np.polyfit(range(10), x, 1)[0])

        # 波动特征
        df_pos['ma_ratio_5_10'] = (df_pos['num'].rolling(5).mean() /
                                   df_pos['num'].rolling(10).mean())

        # 位置特征
        df_pos['pos_ratio'] = df_pos.index / len(df_pos)
        df_pos['last_3_avg'] = df_pos['num'].rolling(3).mean()

        features.append(df_pos.fillna(0).values)

    X = np.hstack(features)

    # 交叉特征
    cross_feat = X[:, :10] * X[:, 10:20]
    X = np.column_stack([X, cross_feat])

    return X, numbers


# ==================== 3. 增强模型架构 ====================
def build_enhanced_model(input_shape):
    """带注意力机制的LSTM+CNN混合模型"""
    inputs = Input(shape=input_shape)

    # 分支1: LSTM处理时序
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = LSTM(64, return_sequences=False)(lstm_out)
    lstm_out = Reshape((1, 64))(lstm_out)

    # 分支2: CNN处理局部模式
    conv_out = Conv1D(64, 3, activation='relu')(inputs)
    conv_out = Conv1D(128, 5, activation='relu')(conv_out)
    conv_out = GlobalMaxPooling1D()(conv_out)
    conv_out = Reshape((1, 128))(conv_out)

    # 合并分支
    merged = concatenate([lstm_out, conv_out], axis=-1)
    merged = Reshape((192,))(merged)

    # 分类层
    dense = Dense(256, activation='relu')(merged)
    dense = Dropout(0.4)(dense)
    dense = Dense(128, activation='relu')(dense)

    # 输出层
    output = Dense(3 * 10)(dense)
    output = Reshape((3, 10))(output)
    output = TimeDistributed(Dense(10, activation='softmax'))(output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(0.00005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==================== 4. 主流程 ====================
def main():
    # 1. 数据加载
    numbers = load_and_validate_data(r"D:\Pythonprojet\lottery_results_all_pages.csv")

    # 2. 特征工程
    X, y = create_enhanced_features(numbers)
    print(f"特征矩阵形状: {X.shape}")

    # 3. 数据标准化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 构建序列数据
    seq_len = 20
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i + seq_len])
        y_seq.append(y[i + seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 5. One-hot编码
    y_onehot = np.zeros((len(y_seq), 3, 10))
    for i in range(3):
        y_onehot[:, i] = np.eye(10)[y_seq[:, i]]

    # 6. 数据集拆分
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_onehot, test_size=0.15, shuffle=True, random_state=42)

    # 7. 构建增强模型
    print("\n=== 模型架构 ===")
    model = build_enhanced_model((seq_len, X_train.shape[2]))
    model.summary()

    # 8. 训练配置
    callbacks = [
        EarlyStopping(patience=50, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=20)
    ]

    # 9. 训练模型
    print("\n=== 开始训练 ===")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    # 10. 保存模型
    model.save('lottery_model.keras')
    print("模型已保存为 lottery_model.keras")


if __name__ == "__main__":
    main()
