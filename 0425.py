import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import re

# 配置参数
CONFIG = {
    "seq_length": 50,  # 模型训练时的固定序列长度（不可修改）
    "batch_size": 128,
    "epochs": 100,
    "data_path": r"D:\Pythonprojet\lottery_results_all_pages.csv",
    "target_digits": 3,
    "validation_split": 0.2,
    "model_path": "lottery_model_0424.keras",
    "top_k": 5,
    "history_length": 60  # 新增：预测时使用多少期历史数据（必须 >= seq_length）
}


class LotteryPredictor:
    def __init__(self, config=None):
        self.CONFIG = CONFIG if config is None else config
        print(f"[Init] 生效配置 history_length = {self.CONFIG['history_length']}")  # 调试点
        # 其余初始化代码...

        self.model = None
        self.data_cache = None
        self.last_update = None

        try:
            if os.path.exists(CONFIG['model_path']):
                print(f"加载已有模型: {CONFIG['model_path']}")
                self.model = load_model(CONFIG['model_path'])
        except Exception as e:
            print(f"加载模型失败: {str(e)}")

    def _parse_number(self, num_str):
        """数字解析方法"""
        try:
            cleaned = re.sub(r'[^\d]', '', str(num_str))
            return int(cleaned) if cleaned else 0
        except:
            return 0

    def _refresh_data(self):
        try:
            if not os.path.exists(CONFIG['data_path']):
                raise FileNotFoundError(f"数据文件不存在: {CONFIG['data_path']}")

            current_mtime = os.path.getmtime(CONFIG['data_path'])
            if self.last_update != current_mtime or self.data_cache is None:
                print("正在更新数据缓存...")
                df = pd.read_csv(CONFIG['data_path'])

                # === 调试输出：验证原始数据 ===
                print("\n[调试] CSV前5行原始数据:")
                print(df.head().to_string())

                assert "开奖号码" in df.columns, "CSV中必须包含'开奖号码'列"
                required_data = max(CONFIG['seq_length'], CONFIG['history_length'])
                assert len(df) > required_data, f"需要至少{required_data + 1}期数据"

                # === 关键修改点 ===
                # 如果CSV第一行已经是最新数据，则不需要倒序
                # df = df.iloc[::-1].reset_index(drop=True)  # 注释掉这行

                df['parsed_numbers'] = df['开奖号码'].apply(
                    lambda x: [self._parse_number(num) for num in str(x).replace('[', '').replace(']', '').split(',')]
                )
                df['parsed_numbers'] = df['parsed_numbers'].apply(
                    lambda x: x[:3] if len(x) >= 3 else x + [0] * (3 - len(x))
                )

                self.data_cache = df['parsed_numbers'].tolist()
                self.last_update = current_mtime

                # === 调试输出：验证处理后数据 ===
                print("\n[调试] 处理后前5条缓存数据（最新在前）:")
                for nums in self.data_cache[:5]:
                    print(nums)
        except Exception as e:
            print(f"数据刷新失败: {str(e)}")
            raise

    def _prepare_training_data(self):
        """准备训练数据（保持时间顺序）"""
        try:
            X, y = [], []
            for i in range(len(self.data_cache) - CONFIG['seq_length']):
                seq = self.data_cache[i:i + CONFIG['seq_length']]
                target = self.data_cache[i + CONFIG['seq_length']]
                X.append(seq)
                y.append(target)

            print(f"生成训练样本: {len(X)}个")
            return np.array(X), np.array(y)

        except Exception as e:
            print(f"准备训练数据失败: {str(e)}")
            raise

    def train(self):
        """训练模型（保持不变）"""
        try:
            self._refresh_data()
            X, y = self._prepare_training_data()

            split_idx = int(len(X) * (1 - CONFIG['validation_split']))
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_val, y_val = X[split_idx:], y[split_idx:]
            print(f"训练集: {len(X_train)} | 验证集: {len(X_val)}")

            if self.model is None:
                print("构建新模型...")
                inputs = Input(shape=(CONFIG['seq_length'], CONFIG['target_digits']))
                x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
                x = Dropout(0.4)(x)
                x = BatchNormalization()(x)
                x = Bidirectional(LSTM(64))(x)
                x = Dropout(0.3)(x)
                x = BatchNormalization()(x)

                outputs = []
                for i in range(CONFIG['target_digits']):
                    branch = Dense(64, activation='relu')(x)
                    branch = Dense(32, activation='relu')(branch)
                    outputs.append(Dense(10, activation='softmax', name=f'digit_{i}')(branch))

                self.model = Model(inputs=inputs, outputs=outputs)
                self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=[['accuracy'] for _ in range(CONFIG['target_digits'])]
                )

            callbacks = [
                EarlyStopping(patience=20, monitor='val_loss'),
                ModelCheckpoint(CONFIG['model_path'], save_best_only=True)
            ]

            print("开始训练...")
            self.model.fit(
                X_train,
                [y_train[:, i] for i in range(3)],
                validation_data=(X_val, [y_val[:, i] for i in range(3)]),
                batch_size=CONFIG['batch_size'],
                epochs=CONFIG['epochs'],
                callbacks=callbacks,
                shuffle=False,
                verbose=1
            )

            self.model.save(CONFIG['model_path'])
            print(f"模型已保存至: {CONFIG['model_path']}")

        except Exception as e:
            print(f"训练失败: {str(e)}")
            raise

    def predict_next(self, history_length=None):
        """
        预测下一期号码
        :param history_length: 可选，指定使用多少期历史数据（必须 >= seq_length）
        """
        try:
            self._refresh_data()
            if self.model is None:
                raise ValueError("未加载模型，请先训练")

            # 确定实际使用的历史期数
            use_length = history_length if history_length is not None else CONFIG['history_length']
            use_length = max(use_length, CONFIG['seq_length'])  # 不能小于模型要求的seq_length

            # 获取历史数据（data_cache已经是倒序，最新数据在前）
            available_data = self.data_cache[:use_length]
            if len(available_data) < use_length:
                raise ValueError(f"需要至少{use_length}期历史数据，当前只有{len(available_data)}期")

            # 从历史数据中提取最后seq_length期用于预测（保持时间连续性）
            predict_data = available_data[:CONFIG['seq_length']]

            # 执行预测
            preds = self.model.predict(np.array([predict_data]), verbose=0)
            predicted_numbers = [np.argmax(preds[i][0]) for i in range(3)]
            probability_dist = []

            for i in range(3):
                top_k = np.argsort(preds[i][0])[-CONFIG['top_k']:][::-1]
                probability_dist.append([(int(num), float(preds[i][0][num])) for num in top_k])

            # 显示结果
            show_length = min(use_length, 10)  # 最多显示10期
            print(f"\n最新{show_length}期历史数据（从新到旧）:")
            for nums in available_data[:show_length]:
                print(nums)

            print(f"\n使用最近{CONFIG['seq_length']}期数据预测下一期号码: {predicted_numbers}")
            for i in range(3):
                print(f"第{i + 1}位数字概率Top{CONFIG['top_k']}:")
                for num, prob in probability_dist[i]:
                    print(f"  {num}: {prob * 100:.2f}%")

            return predicted_numbers, probability_dist

        except Exception as e:
            print(f"预测失败: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        predictor = LotteryPredictor()

        if not os.path.exists(CONFIG['model_path']):
            print("未检测到已有模型，开始训练...")
            predictor.train()

        print("\n开始预测...")
        # 示例：使用20期历史数据进行预测（会自动截取最新的10期给模型）
        predictor.predict_next(history_length=60)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
