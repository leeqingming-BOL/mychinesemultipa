from datasets import load_dataset, load_metric, Audio, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
import argparse
import pandas as pd
import os
import multiprocess

from data_utils import filter_low_quality, downsampling

def extract_all_chars_ipa(batch: dict) -> dict:
    # 创建基于IPA音素的词汇表
    all_text = " ".join(batch["ipa"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset_ipa(batch: dict) -> dict:
    audio = batch["audio"]

    # batched output is unbatched
    batch["input_values"] = processor_ipa(audio["array"],
                                          sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor_ipa.as_target_processor():
        batch["labels"] = processor_ipa(batch["ipa"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
                )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def remove_long_data(dataset, max_seconds=6):
    # 移除超过指定长度的音频数据
    dftest = dataset.to_pandas()
    dftest['len'] = dftest['input_values'].apply(len)
    # 已重采样到16khz，移除大于max_seconds的数据
    maxLength = max_seconds * 16000 
    dftest = dftest[dftest['len'] < maxLength]
    dftest = dftest.drop(columns=['len'])
    dataset = dataset.from_pandas(dftest)
    del dftest
    return dataset

def remove_space(batch: dict) -> dict:
    # 移除IPA中的空格
    ipa = batch["ipa"]
    ipa = ipa.split()
    ipa = "".join(ipa)
    batch["ipa"] = ipa
    return batch

if __name__ == "__main__":
    # 简化的命令行参数
    parser = argparse.ArgumentParser(description="中文语音到IPA转录训练")

    parser.add_argument("-tr", "--train_samples", type=int, default=1000,
                        help="指定用于训练的样本数量")
    parser.add_argument("-te", "--test_samples", type=int, default=200,
                        help="指定用于测试的样本数量")
    parser.add_argument("-qf", "--quality_filter", type=bool, default=True,
                        help="是否过滤质量低的音频(至少有1个负面评价)")
    parser.add_argument("-s", "--suffix", type=str, default="",
                        help="模型名称后缀，用于标识不同的训练版本")
    parser.add_argument("-ns", "--no_space", type=bool, default=False,
                        help="设置为True以从训练和测试数据中删除空格")
    parser.add_argument("-v", "--vocab_file", type=str, default="vocab_zh.json",
                        help="指定要创建的词汇文件名")
    parser.add_argument("-dd", "--data_dir", type=str, default="data_new/",
                        help="指定训练/验证数据文件的目录路径")
    parser.add_argument("-ds", "--dataset", type=str, default="mozilla-foundation/common_voice_11_0",
                        help="指定数据集名称")
    parser.add_argument("-e", "--num_train_epochs", type=int, default=30,
                        help="指定训练轮数，默认为30")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="指定预处理使用的CPU数量，默认为8")
    args = parser.parse_args()
    suffix = args.suffix
    
    # 加载中文数据
    print("加载中文数据...")
    stats_file = "stats_train_valid_zh_{}.txt".format(suffix)
    with open(stats_file, "w") as f:
        f.write("lang train valid\n")
    
    # 直接从Preprocessors获取中文数据
    from data_utils import Preprocessors
    train, valid = Preprocessors.chinese(args.train_samples, args.test_samples, args.quality_filter)
    
    # 转换为audio格式
    train = train.cast_column("audio", Audio())
    valid = valid.cast_column("audio", Audio())
    
    print("中文训练样本数量: {}".format(len(train)))
    print("中文验证样本数量: {}".format(len(valid)))
    
    with open(stats_file, "a") as f:
        f.write("zh " + str(len(train)) + " " + str(len(valid)) + "\n")
    
    # 移除不必要的列
    print("移除不必要的列...")
    train = train.remove_columns([col for col in [
        "accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes",
        "speaker_id", "chapter_id", "id"
        ] if col in train.column_names])
    valid = valid.remove_columns([col for col in [
        "accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes",
        "speaker_id", "chapter_id", "id"
        ] if col in valid.column_names])
    print("不必要的列已移除。数据预览:")
    print(train[0])
    
    # 如果指定，则移除空格
    if args.no_space:
        print("移除空格...")
        train = train.map(remove_space)
        valid = valid.map(remove_space)
        print("完成空格移除")
    
    # 打乱数据集
    print("打乱数据集...")
    train = train.shuffle(seed=42)
    valid = valid.shuffle(seed=35)
    print("数据集已打乱")

    # 创建词汇表
    print("创建词汇表...")
    vocab_train_ipa = train.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=train.column_names
    )
    vocab_valid_ipa = valid.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=valid.column_names
    )
    vocab_list_ipa = list(
        set(vocab_train_ipa["vocab"][0]) | set(vocab_valid_ipa["vocab"][0])
    )
    # 添加多字母IPAs和其他IPAs
    with open("full_vocab_ipa.txt", "r") as f:
        lines = f.readlines()
        ipa_all = set([l.strip() for l in lines])
    vocab_list_ipa = set(vocab_list_ipa) | ipa_all
    vocab_list_ipa = list(vocab_list_ipa)
    vocab_dict_ipa = {v: k for k, v in enumerate(vocab_list_ipa)}

    print("词汇表已创建。详情:")
    print("vocab_dict_ipa大小: {}".format(len(vocab_dict_ipa)))

    # 添加[UNK], [PAD]
    print("添加[UNK]和[PAD]...")
    vocab_dict_ipa["[UNK]"] = len(vocab_dict_ipa)
    vocab_dict_ipa["[PAD]"] = len(vocab_dict_ipa)
    print("[UNK]和[PAD]已添加")

    # 写入词汇json文件
    print("写入词汇json文件...")
    with open(args.vocab_file, 'w') as vocab_file_ipa:
        json.dump(vocab_dict_ipa, vocab_file_ipa)
    print("词汇json文件已创建")

    # 创建分词器
    print("创建分词器...")
    tokenizer_ipa = Wav2Vec2CTCTokenizer("./{}".format(args.vocab_file),
                                         unk_token="[UNK]",
                                         pad_token="[PAD]",
                                         word_delimiter_token="|")
    print("分词器已创建") 

    # 创建特征提取器
    print("创建特征提取器...")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                               sampling_rate=16_000,
                                               padding_value=0.0,
                                               do_normalize=True,
                                               return_attention_mask=True)
    print("特征提取器已创建") 

    # 定义处理器
    print("创建处理器...")
    processor_ipa = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                    tokenizer=tokenizer_ipa)
    print("处理器已创建") 

    # 设置采样率为16,000Hz
    print("调整采样率为16,000Hz...")
    train = train.cast_column("audio", Audio(sampling_rate=16_000))
    valid = valid.cast_column("audio", Audio(sampling_rate=16_000))
    print("采样率调整完成")

    # 预处理数据集
    print("预处理数据集...")
    train = train.map(
        prepare_dataset_ipa,
        remove_columns=train.column_names,
        num_proc=args.num_proc
    )
    valid = valid.map(
        prepare_dataset_ipa,
        remove_columns=valid.column_names,
        num_proc=args.num_proc
    )
    
    # 移除超过6秒的音频文件
    print("移除超过6秒的音频文件...")
    train = remove_long_data(train)
    valid = remove_long_data(valid)
    print("要训练和测试的数据集大小:")
    print("训练集:", len(train))
    print("验证集:", len(valid))
    print("预处理完成")

    # 创建数据收集器
    print("创建数据收集器")
    data_collator = DataCollatorCTCWithPadding(processor=processor_ipa, padding=True)
    print("数据收集器已创建")
    
    # 定义模型
    print("定义模型...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor_ipa.tokenizer.pad_token_id,
        vocab_size=len(processor_ipa.tokenizer)
        )
    print("模型已定义")

    # 冻结特征提取器，使其在微调过程中不会更改
    print("冻结特征提取器...") 
    model.freeze_feature_extractor()
    print("特征提取器已冻结")

    # 设置输出目录
    output_dir = "./wav2vec2-large-xlsr-zh-ipa"
    if suffix:
        output_dir += suffix
        
    # 开始训练
    print("开始训练...") 
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor_ipa.feature_extractor,
        )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()
    # trainer.push_to_hub(repo_name="wav2vec2-ipa-zh")
