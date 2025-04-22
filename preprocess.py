from datasets import load_dataset, Audio, Dataset
from argparse import ArgumentParser
import json
import re
import time
import pandas as pd
import os
import shutil

import sys
sys.path.insert(0, "./converter")

from chinese_to_ipa import Chinese2IPA

parser = ArgumentParser(description="创建中文语音到IPA转换数据集。")
parser.add_argument("--output_dir", type=str, default="data_new",
                    help="指定预处理数据的输出目录。")
parser.add_argument("--num_proc", type=int, default=1,
                    help="指定用于多处理的核心数量。默认设置为1（无多处理）。")
parser.add_argument("--clear_cache", action="store_true",
                    help="如果要在加载后清除数据集缓存以防止内存崩溃，请使用此选项。")
parser.add_argument("--cache_dir", type=str,
                    help="如果选择清除缓存，请指定缓存目录的路径。")
args = parser.parse_args()
if args.clear_cache and args.cache_dir is None:
    print("警告：启用了清除缓存但未指定缓存目录路径。")

def transliterate(sample: dict):
    """将中文文本转换为IPA音标"""
    sent = sample["sentence"]
    ipa = Chinese2IPA.chinese_generate_ipa(sent)
    sample["ipa"] = "".join(ipa.split())
    return sample

def remove_audio_column(dataset) -> Dataset:
    """移除数据集的[audio]列，以便可以保存为json。
    'array'会导致`OverflowError: Unsupported UTF-8 sequence length when encoding string`，
    所以需要移除它。这个列将在训练时通过直接下载的数据恢复。
    """
    if "audio" in dataset.column_names:
        dataset = dataset.remove_columns(["audio"])
    return dataset
    
# 数据集处理
if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    stats_file = "{}/presave_zh_trainvalid_stats.tsv".format(args.output_dir)
    with open(stats_file, "w") as f:
        f.write("数据集\t训练样本数\t验证样本数\t处理时间\n")
    
    start = time.time()
    
    print("正在加载中文数据集...")
    try:
        train = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN",
                            split="train")
        valid = load_dataset("mozilla-foundation/common_voice_11_0", "zh-CN",
                            split="validation")
        
        print(f"成功加载中文数据集，训练集大小: {len(train)}，验证集大小: {len(valid)}")
    except Exception as e:
        print(f"加载中文数据集时出错: {e}")
        exit(1)

    # 移除音频列
    print("移除音频列...")
    train = remove_audio_column(train)
    valid = remove_audio_column(valid)

    # 转换为IPA
    print("将中文文本转换为IPA...")
    train = train.map(transliterate, 
                      num_proc=args.num_proc, 
                      desc="处理训练集")
    valid = valid.map(transliterate, 
                      num_proc=args.num_proc, 
                      desc="处理验证集")

    # 导出为json
    print("保存处理后的数据集...")
    train.to_json("{}/zh_train.json".format(args.output_dir))
    valid.to_json("{}/zh_valid.json".format(args.output_dir))

    end = time.time()
    duration = end - start
    print(f"中文数据处理完成！耗时: {duration:.2f}秒")
    
    with open(stats_file, "a") as f:
        f.write("中文\t{}\t{}\t{:.2f}\n".format(len(train), len(valid), duration))

    # 清除缓存
    if args.clear_cache and args.cache_dir:
        print("正在清除缓存...")
        shutil.rmtree(args.cache_dir, ignore_errors=True)
        print("缓存已清除")
