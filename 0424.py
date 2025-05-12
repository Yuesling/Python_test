import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from typing import List, Tuple
from collections import defaultdict
from math import log
import random


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """安全的除法运算（避免除零错误）"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = 0
    return result


def create_enhanced_features(numbers: np.ndarray, df: pd.DataFrame = None) -> np.ndarray:
    """生成88维特征向量"""
    numbers = numbers.astype(np.float32)
    features = []

    # 基础位置特征
    for pos in range(3):
        pos_data = numbers[:, pos]
        pos_features = []

        # 滑动窗口统计
        for window in [5, 10, 15]:
            rolling = pd.Series(pos_data).rolling(window)
            pos_features.extend([
                rolling.mean().values[-1],
                rolling.std().values[-1],
                rolling.max().values[-1],
                rolling.min().values[-1]
            ])
        features.extend(pos_features)

    # 高级全局特征
    features.extend([
        numbers.mean(), numbers.std(),
        numbers.max(), numbers.min(),
        (numbers % 2 == 1).mean(),
        (numbers > 5).mean(),
        len(set(numbers.flatten())) / 10
    ])

    # 填充至88维
    features = np.pad(features, (0, max(0, 88 - len(features)))[:88])
    return np.array(features, dtype=np.float32)


def prepare_input_data(data_path: str, seq_len: int = 20, num_samples: int = 3) -> np.ndarray:
    """准备输入数据"""
    try:
        df = pd.read_csv(data_path)
        numbers = np.array([np.array(ast.literal_eval(x), dtype=np.int32)
                            for x in df['开奖号码']])

        # 计算数字历史频率（新增）
        df['数字频率'] = df['开奖号码'].apply(lambda x:
                                              [ast.literal_eval(x).count(i) for i in range(10)])
        freq_features = np.array(df['数字频率'].tolist())

        X = []
        for i in range(seq_len, len(numbers)):
            window = numbers[i - seq_len:i]
            features = create_enhanced_features(window)

            # 加入历史频率特征（新增）
            freq_window = freq_features[i - seq_len:i]
            freq_feat = freq_window.mean(axis=0)
            features = np.concatenate([features, freq_feat])

            X.append(features[:88])  # 确保88维

        X = np.array(X)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        input_sequences = []
        for i in range(num_samples):
            seq = X_scaled[-seq_len - i: -i if i != 0 else None]
            input_sequences.append(seq)

        return np.array(input_sequences, dtype=np.float32)

    except Exception as e:
        print(f"数据准备错误: {str(e)}")
        raise


def advanced_prediction(
        model: tf.keras.Model,
        input_data: np.ndarray,
        num_results: int = 9,
        temp: float = 0.5,
        min_sum: int = 10,
        max_sum: int = 20,
        freq_penalty: float = 1.2,
        pos_penalty: float = 0.8,
        custom_weights: np.ndarray = None
) -> List[Tuple[List[int], float]]:
    """高级预测函数"""
    # 获取历史趋势数据
    history = input_data[:, :, :10]
    trend_weights = 1 + np.mean(history, axis=(0, 1))[:10]

    if custom_weights is None:
        custom_weights = np.ones(10)

    raw_probs = model.predict(input_data, verbose=0)

    # 三重惩罚机制
    adj_probs = np.zeros_like(raw_probs)
    for i in range(3):
        for j in range(10):
            base = np.power(raw_probs[:, i, j], 1 / temp)
            freq_pen = np.exp(-np.sum(raw_probs[:, :, j]) * freq_penalty)
            pos_pen = np.exp(-np.sum(raw_probs[:, i, :]) * pos_penalty)
            adj_probs[:, i, j] = base * freq_pen * pos_pen * trend_weights[j] * custom_weights[j]

    adj_probs /= np.sum(adj_probs, axis=-1, keepdims=True)

    # 蒙特卡洛采样
    candidates = []
    for _ in range(5000):
        nums = []
        for pos in range(3):
            probs = adj_probs[0, pos]
            nums.append(np.random.choice(10, p=probs))

        if min_sum <= sum(nums) <= max_sum:
            prob = 1.0
            for pos, num in enumerate(nums):
                prob *= adj_probs[0, pos, num]

            # 多样性奖励
            unique_nums = len(set(nums))
            prob *= (1 + 0.1 * unique_nums)

            candidates.append((nums, prob))

    # 结果处理
    candidates.sort(key=lambda x: -x[1])
    unique = defaultdict(float)
    for nums, prob in candidates:
        unique[tuple(nums)] += prob

    total = sum(unique.values())
    results = [(list(k), v / total) for k, v in unique.items()]

    # 多样性选择
    final_results = []
    used_numbers = set()
    for nums, prob in sorted(results, key=lambda x: -x[1]):
        if len(final_results) >= num_results:
            break

        new_nums = set(nums) - used_numbers
        if new_nums or not used_numbers:
            # 调整特殊组合概率
            odd_count = sum(n % 2 for n in nums)
            if odd_count == 0:  # 全偶
                prob *= 0.8
            elif odd_count == 3:  # 全奇
                prob *= 0.9

            final_results.append((nums, prob))
            used_numbers.update(nums)

    # 最终归一化
    final_total = sum(p for _, p in final_results)
    return [(nums, p / final_total) for nums, p in final_results]


def format_results(results: List[Tuple[List[int], float]]) -> None:
    """美观格式化输出"""
    print("\n🔮 高级预测结果:")
    print("=" * 60)
    print(f"{'组合':<18} | {'概率':<10} | {'和值':<6} | {'奇偶比':<10} | 特性")
    print("-" * 60)

    for nums, prob in results:
        odd = sum(n % 2 for n in nums)
        props = []
        if len(set(nums)) < 3: props.append("有重复")
        if sum(nums) < 12: props.append("和小")
        if sum(nums) > 18: props.append("和大")
        if odd == 0: props.append("全偶")
        if odd == 3: props.append("全奇")

        print(
            f"{' '.join(f'{n:02d}' for n in nums):<18} | "
            f"{prob:.2%}    | "
            f"{sum(nums):<6} | "
            f"{odd}奇{3 - odd}偶    | "
            f"{', '.join(props) if props else '一般'}"
        )


def analyze_results(results: List[Tuple[List[int], float]]):
    """结果分析报告"""
    print("\n📊 预测结果分析报告:")
    print("=" * 60)

    # 数字频率统计
    num_dist = defaultdict(int)
    for nums, _ in results:
        for n in nums:
            num_dist[n] += 1

    print(f"数字分布: {dict(sorted(num_dist.items()))}")
    print(f"平均概率: {sum(p for _, p in results) / len(results):.2%}")
    print(f"和值范围: {min(sum(nums) for nums, _ in results)}-{max(sum(nums) for nums, _ in results)}")

    # 组合类型分析
    types = {
        '有重复': sum(1 for nums, _ in results if len(set(nums)) < 3),
        '全奇': sum(1 for nums, _ in results if all(n % 2 == 1 for n in nums)),
        '全偶': sum(1 for nums, _ in results if all(n % 2 == 0 for n in nums)),
        '大和(>17)': sum(1 for nums, _ in results if sum(nums) > 17),
        '小和(<13)': sum(1 for nums, _ in results if sum(nums) < 13)
    }
    print("组合特性:", types)


def validate_results(results: List[Tuple[List[int], float]]):
    """结果验证"""
    try:
        assert len(set(tuple(x[0]) for x in results)) == len(results)
        assert all(10 <= sum(x[0]) <= 20 for x in results)
        print("✅ 结果验证通过")
    except AssertionError as e:
        print(f"❌ 结果验证失败: {str(e)}")


if __name__ == "__main__":
    # 加载模型
    model = tf.keras.models.load_model('lottery_model.keras')
    print(f"✅ 模型加载成功 (输入: {model.input_shape} 输出: {model.output_shape})")

    # 自定义数字权重（可选）
    custom_weights = np.array([
        1.0,  # 0
        1.0,  # 1
        1.0,  # 2
        1.0,  # 3
        1.0,  # 4 (降低出现概率)
        1.0,  # 5 (提高出现概率)
        1.0,  # 6
        1.0,  # 7
        1.0,  # 8
        1.0   # 9
    ])

    # 准备数据
    data_path = r"D:\Pythonprojet\lottery_results_all_pages.csv"
    try:
        input_data = prepare_input_data(data_path)
        print(f"✅ 输入数据验证: 形状={input_data.shape} 范围=[{input_data.min():.2f}, {input_data.max():.2f}]")

        # 执行高级预测
        results = advanced_prediction(
            model,
            input_data,
            num_results=9,
            temp=0.5,
            min_sum=10,
            max_sum=20,
            freq_penalty=1.2,
            pos_penalty=0.8,
            custom_weights=custom_weights
        )

        # 显示和分析结果
        format_results(results)
        analyze_results(results)
        validate_results(results)

        # 保存结果
        result_df = pd.DataFrame({
            '数字1': [r[0][0] for r in results],
            '数字2': [r[0][1] for r in results],
            '数字3': [r[0][2] for r in results],
            '概率': [r[1] for r in results],
            '和值': [sum(r[0]) for r in results],
            '奇偶比': [f"{sum(n % 2 for n in r[0])}奇{3 - sum(n % 2 for n in r[0])}偶" for r in results],
            '特性': [', '.join([
                '有重复' if len(set(r[0])) < 3 else '',
                '和小' if sum(r[0]) < 12 else '',
                '和大' if sum(r[0]) > 18 else '',
                '全偶' if all(n % 2 == 0 for n in r[0]) else '',
                '全奇' if all(n % 2 == 1 for n in r[0]) else ''
            ]).strip(', ') or '一般' for r in results]
        })
        result_df.to_csv('advanced_predictions.csv', index=False)
        print("\n💾 预测结果已保存到 advanced_predictions.csv")

    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
