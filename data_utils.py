from datasets import load_dataset, Dataset
from chinese_to_ipa import Chinese2IPA
import random
import re

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
    def chinese(cls, train_samples, test_samples, quality_filter=False):
        """
        处理中文数据集
        
        参数:
        train_samples: 训练集样本数量
        test_samples: 测试集样本数量
        quality_filter: 是否过滤低质量样本
        
        返回:
        训练集和验证集
        """
        # 加载数据集
        print("加载中文数据集...")
        zh_train = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="train")
        zh_test = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN", split="validation")
        
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
