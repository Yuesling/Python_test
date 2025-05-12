import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Input, Concatenate, MultiHeadAttention,
                                     LayerNormalization)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard, LearningRateScheduler)
import matplotlib.pyplot as plt
from collections import Counter
import warnings

# 配置参数
CONFIG = {
    "data_path": r"D:\Pythonprojet\lottery_results_all_pages.csv",
    "seq_length": 50,
    "batch_size": 128,
    "epochs": 300,
    "patience": 40,
    "validation_split": 0.15,
    "model_save_path": "lottery_model_final.keras",
    "log_dir": "./logs",
    "num_heads": 4,
    "key_dim": 64,
    "ff_dim": 256
}

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')


def safe_convert(x):
    try:
        numbers = ast.literal_eval(x) if isinstance(x, str) else x
        return np.array(numbers, dtype=np.int32)
    except:
        return np.array([0, 0, 0], dtype=np.int32)


def calculate_sample_weights(y):
    """计算样本权重"""
    y_flat = y.reshape(-1)
    counts = Counter(y_flat)
    total = sum(counts.values())
    weight_dict = {k: total / (v * len(counts)) for k, v in counts.items()}

    sample_weights = np.zeros(len(y))
    for i in range(len(y)):
        sample_weights[i] = np.mean([weight_dict[n] for n in y[i]])
    return sample_weights


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return tf.cast(pos_encoding[tf.newaxis, ...], tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    res = x + inputs
    x = LayerNormalization(epsilon=1e-6)(res)

    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Dropout(dropout)(x)
    return LayerNormalization(epsilon=1e-6)(x + res)


def build_hybrid_model(input_shape, freq_shape, output_shape):
    main_input = Input(shape=input_shape, name='main_input')
    freq_input = Input(shape=freq_shape, name='freq_input')

    # LSTM分支
    lstm_out = LSTM(256, return_sequences=True)(main_input)
    lstm_out = Dropout(0.2)(lstm_out)

    # Transformer分支
    pos_encoding = PositionalEncoding(input_shape[0], input_shape[1])(main_input)
    trans_out = transformer_encoder(
        pos_encoding,
        head_size=CONFIG["key_dim"],
        num_heads=CONFIG["num_heads"],
        ff_dim=CONFIG["ff_dim"]
    )
    trans_out = tf.keras.layers.GlobalAveragePooling1D()(trans_out)

    # 合并特征
    merged = Concatenate()([lstm_out[:, -1, :], trans_out, freq_input])

    # 共享层
    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 独立输出层
    outputs = []
    for i in range(output_shape[0]):
        y = Dense(128, activation='relu')(x)
        y = Dense(output_shape[-1], activation='softmax', name=f'pos_{i + 1}')(y)
        outputs.append(y)

    return Model(inputs=[main_input, freq_input], outputs=outputs)


def lr_scheduler(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch < 150:
        return lr * 0.9
    else:
        return max(lr * 0.95, 1e-5)


def load_and_preprocess_data():
    df = pd.read_csv(CONFIG["data_path"])
    print(f"原始数据量: {len(df)}条")

    df["numbers"] = df["开奖号码"].apply(safe_convert)
    df = df[df["numbers"].apply(lambda x: x.shape == (3,))]
    print(f"有效数据量: {len(df)}条")

    X_main, X_freq, y = [], [], []
    for i in range(CONFIG["seq_length"], len(df)):
        seq = np.vstack(df["numbers"][i - CONFIG["seq_length"]:i].values)
        target = df["numbers"][i]

        # 主特征
        main_features = []
        for pos in range(3):
            pos_data = seq[:, pos].astype(np.float32)

            stats = [
                np.mean(pos_data), np.std(pos_data),
                np.min(pos_data), np.max(pos_data),
                np.median(pos_data),
                np.percentile(pos_data, 25),
                np.percentile(pos_data, 75),
                np.polyfit(range(len(pos_data)), pos_data, 1)[0],
                np.fft.fft(pos_data).real[1]
            ]

            for window in [5, 10, 15, 20]:
                rolling = pd.Series(pos_data).rolling(window)
                stats.extend([
                    rolling.mean().iloc[-1],
                    rolling.std().iloc[-1],
                    rolling.min().iloc[-1],
                    rolling.max().iloc[-1]
                ])
            main_features.extend(stats)

        sums = seq.sum(axis=1)
        main_features.extend([
            np.mean(seq), np.std(seq),
            len(np.unique(seq)) / (CONFIG["seq_length"] * 3),
            np.mean(seq % 2 == 1),
            np.mean(sums), np.std(sums),
            np.sum(sums > 15) / len(sums)
        ])

        # 频率特征
        freq_features = []
        for n in range(10):
            freq_features.append(np.mean(seq == n))
        for pos in range(3):
            pos_counts = np.bincount(seq[:, pos], minlength=10) / len(seq)
            freq_features.extend(pos_counts)

        X_main.append(np.array(main_features, dtype=np.float32))
        X_freq.append(np.array(freq_features, dtype=np.float32))
        y.append(target)

    main_scaler = RobustScaler()
    X_main = main_scaler.fit_transform(np.array(X_main))

    freq_scaler = MinMaxScaler()
    X_freq = freq_scaler.fit_transform(np.array(X_freq))

    X_main = X_main.reshape(-1, 1, X_main.shape[1])
    y = np.array(y, dtype=np.int32)

    y_onehot = np.zeros((len(y), 3, 10), dtype=np.float32)
    for i in range(len(y)):
        for pos in range(3):
            y_onehot[i, pos, y[i, pos]] = 1

    return X_main, X_freq, y_onehot, main_scaler, freq_scaler


# ... (前面的导入和配置保持不变)

def train_model():
    X_main, X_freq, y, main_scaler, freq_scaler = load_and_preprocess_data()

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, val_index in tscv.split(X_main):
        X_train_main, X_val_main = X_main[train_index], X_main[val_index]
        X_train_freq, X_val_freq = X_freq[train_index], X_freq[val_index]
        y_train, y_val = y[train_index], y[val_index]

    # 计算样本权重
    sample_weights = calculate_sample_weights(
        np.argmax(y_train.reshape(-1, 10), axis=1).reshape(-1, 3)
    )

    model = build_hybrid_model(
        input_shape=X_train_main.shape[1:],
        freq_shape=X_train_freq.shape[1:],
        output_shape=y_train.shape[1:]
    )
    model.summary()

    # 为每个输出定义损失函数
    losses = {
        'pos_1': 'categorical_crossentropy',
        'pos_2': 'categorical_crossentropy',
        'pos_3': 'categorical_crossentropy'
    }

    # 定义损失权重
    loss_weights = {
        'pos_1': 1.0,
        'pos_2': 1.0,
        'pos_3': 1.0
    }

    # 为每个输出定义评估指标
    metrics = {
        'pos_1': ['accuracy'],
        'pos_2': ['accuracy'],
        'pos_3': ['accuracy']
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics  # 使用字典格式指定每个输出的指标
    )

    callbacks = [
        EarlyStopping(
            monitor='val_pos_1_accuracy',
            patience=CONFIG["patience"],
            mode='max',
            restore_best_weights=True
        ),
        ModelCheckpoint(
            CONFIG["model_save_path"],
            monitor='val_pos_1_accuracy',
            save_best_only=True
        ),
        TensorBoard(log_dir=CONFIG["log_dir"]),
        LearningRateScheduler(lr_scheduler)
    ]

    history = model.fit(
        x=[X_train_main, X_train_freq],
        y=[y_train[:, 0, :], y_train[:, 1, :], y_train[:, 2, :]],
        validation_data=(
            [X_val_main, X_val_freq],
            [y_val[:, 0, :], y_val[:, 1, :], y_val[:, 2, :]]
        ),
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        sample_weight=sample_weights,
        verbose=1
    )

    plt.figure(figsize=(18, 6))
    for i, pos in enumerate(['pos_1', 'pos_2', 'pos_3']):
        plt.subplot(1, 3, i + 1)
        plt.plot(history.history[f'{pos}_accuracy'], label=f'Train {pos}')
        plt.plot(history.history[f'val_{pos}_accuracy'], label=f'Val {pos}')
        plt.title(f'Position {i + 1} Accuracy')
        plt.legend()
    plt.savefig('training_metrics.png')
    plt.show()

    model.save(CONFIG["model_save_path"])
    print(f"模型已保存到 {CONFIG['model_save_path']}")

    best_acc = max(history.history['val_pos_1_accuracy'])
    print(f"最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    train_model()
