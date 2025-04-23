from datasets import load_dataset, Dataset, Audio
from converter.chinese_to_ipa import Chinese2IPA
import random
import re
import pandas as pd
import os

def selection(dataset, selectsize: int):
    """
    从数据集中选择指定数量的样本
    """
    trainsize = len(dataset)
    samples = sampling(trainsize, selectsize)
    selected = dataset.select(samples)
    return selected

def filter_low_quality(dataset):
    """
    过滤掉低质量的音频样本（有负面评价的样本）
    """
    dataset = dataset.filter(lambda batch: batch["down_votes"] == 0)
    return dataset

def convert_chinese_to_ipa(batch: dict) -> dict:
    """
    将中文句子转换为IPA音标
    注意：chinese_generate_ipa也会处理标点符号的移除
    """
    sent = batch["sentence"]
    batch["ipa"] = Chinese2IPA.chinese_generate_ipa(sent)
    return batch

def sampling(size: int, n: int) -> list:
    """
    从指定大小的数据集中随机采样n个样本的索引
    
    参数:
    size: 原始数据集的长度（样本数）
    n: 想要获取的样本数量
    
    输出是一个整数列表（数据集的索引），没有重复。
    在训练多语言模型并预计耗时时，使用此函数减少数据集的样本大小。
    """
    numlist = [i for i in range(size)]
    random_samples = random.sample(numlist, n)
    return random_samples

def downsampling(dataset: Dataset, samples: int):
    """
    下采样数据集到指定数量的样本
    
    如果数据集大小小于请求的样本数，则返回整个数据集
    """
    size = len(dataset)
    if size == None or size < samples:
        samples = size
    dataset = selection(dataset, samples)
    return dataset

class Preprocessors:
    """
    数据预处理器类，包含处理不同语言的方法
    现在简化为只处理中文
    """
    
    @classmethod
    def chinese(cls, train_samples, test_samples, quality_filter=False, data_dir="zh-CN"):
        """
        处理中文数据集
        
        参数:
        train_samples: 训练集样本数量
        test_samples: 测试集样本数量
        quality_filter: 是否过滤低质量样本
        data_dir: 本地数据集目录路径
        
        返回:
        训练集和验证集
        """
        # 加载本地数据集
        print("从本地加载中文数据集...")
        
        # 构建完整路径
        train_file = os.path.join(data_dir, "train.tsv")
        test_file = os.path.join(data_dir, "dev.tsv")  # dev作为验证集
        clips_dir = os.path.join(data_dir, "clips")
        
        # 检查文件是否存在
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            raise FileNotFoundError(f"未找到所需的数据文件。请确保{train_file}和{test_file}存在。")
        
        # 从TSV文件加载数据
        train_df = pd.read_csv(train_file, sep='\t')
        test_df = pd.read_csv(test_file, sep='\t')
        
        # 添加音频文件路径
        train_df['audio'] = train_df['path'].apply(lambda x: os.path.join(clips_dir, x))
        test_df['audio'] = test_df['path'].apply(lambda x: os.path.join(clips_dir, x))
        
        # 将DataFrame转换为Dataset
        zh_train = Dataset.from_pandas(train_df)
        zh_test = Dataset.from_pandas(test_df)
        
        # 设置音频列
        zh_train = zh_train.cast_column("audio", Audio(sampling_rate=16000))
        zh_test = zh_test.cast_column("audio", Audio(sampling_rate=16000))
        
        # 过滤低质量样本
        if quality_filter:
            print("过滤低质量音频样本...")
            zh_train = filter_low_quality(zh_train)
            zh_test = filter_low_quality(zh_test)
        
        # 下采样（如果需要的话）
        print(f"下采样到训练集{train_samples}个样本，测试集{test_samples}个样本...")
        zh_train = downsampling(zh_train, train_samples)
        zh_test = downsampling(zh_test, test_samples)
        
        # 转换为IPA
        print("将中文文本转换为IPA音标...")
        zh_train = zh_train.map(convert_chinese_to_ipa)
        zh_test = zh_test.map(convert_chinese_to_ipa)
        
        print(f"中文数据处理完成，训练集大小：{len(zh_train)}，测试集大小：{len(zh_test)}")
        return zh_train, zh_test
