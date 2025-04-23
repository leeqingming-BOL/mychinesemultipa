import os
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import utils

# 1. 加载训练好的模型和处理器
model_path = "wav2vec2-large-xlsr-zh-ipa-zh-ipa1000"  # 模型路径

# 检查模型目录是否包含tokenizer文件
if not os.path.exists(os.path.join(model_path, "vocab.json")):
    print("模型目录中缺少tokenizer文件，正在使用vocab_zh.json重建...")
    
    # 使用vocab_zh.json创建tokenizer
    tokenizer = Wav2Vec2CTCTokenizer("vocab_zh.json",
                                     unk_token="[UNK]",
                                     pad_token="[PAD]",
                                     word_delimiter_token="|")
    
    # 保存tokenizer到模型目录
    tokenizer.save_pretrained(model_path)
    print("已成功将tokenizer保存到模型目录")

# 创建特征提取器
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                           sampling_rate=16000,
                                           padding_value=0.0,
                                           do_normalize=True,
                                           return_attention_mask=True)

# 创建处理器
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# 加载模型
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

# 2. 准备测试数据
from data_utils import Preprocessors
print("正在加载测试数据...")
_, test_dataset = Preprocessors.chinese(1000, 200, quality_filter=True)  # 调整测试样本数量
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
print(f"加载了 {len(test_dataset)} 个测试样本")

# 3. 创建预测函数
def predict(batch):
    audio = batch["audio"]
    # 预处理音频
    input_values = processor(audio["array"], 
                            sampling_rate=audio["sampling_rate"],
                            return_tensors="pt").input_values
    
    # 进行预测
    with torch.no_grad():
        logits = model(input_values).logits
    
    # 解码预测结果
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # 保存预测结果
    batch["predicted_ipa"] = transcription
    return batch

# 4. 应用预测到测试数据
print("开始模型预测...")
test_dataset = test_dataset.map(predict)
print("预测完成")

# 5. 加载特征表
print("加载IPA特征表...")
feature_df = pd.read_csv("ipa_bases.csv", index_col=0)
print(f"已加载特征表，包含 {len(feature_df)} 个IPA音标特征")

# 自定义字符错误率计算函数，避免使用evaluate.load
def compute_cer(pred, ref):
    # 使用已有的levenshteinDistanceDP计算编辑距离
    ld = utils.levenshteinDistanceDP(pred, ref)
    # 计算字符错误率
    return ld / len(ref) if len(ref) > 0 else 0

# 6. 计算评估指标
print("计算评估指标...")
results = []
errors = 0
for idx, item in enumerate(test_dataset):
    try:
        pred = item["predicted_ipa"]
        gold = item["ipa"]
        
        print(f"样本 {idx+1}: 预测={pred}, 实际={gold}")
        
        # 避免evaluate.load引起的问题
        pred = pred.replace("g", "ɡ")  # 统一Unicode字符
        gold = gold.replace("g", "ɡ")
        
        # 计算Levenshtein距离
        ld = utils.levenshteinDistanceDP(pred, gold)
        
        # 计算字符错误率
        cer = compute_cer(pred, gold)
        
        # 计算其他指标
        df_pred = utils.preprocessing_combine(pred, feature_df)
        df_gold = utils.preprocessing_combine(gold, feature_df)
        per = utils.phoneme_error_rate(df_pred, df_gold)
        
        # Levenshtein Phone Distance
        lphd = utils.LPhD_combined(df_pred, df_gold)
        
        # Feature-weighted Phone Error Rate
        fper = lphd / df_gold.shape[0] if df_gold.shape[0] > 0 else 0
        
        metrics = {
            "Levenshtein Distance": ld,
            "Character Error Rate": cer,
            "Phoneme Error Rate": per,
            "Levenshtein Phone Distance": lphd,
            "Feature-weighted Phone Error Rate": fper
        }
        
        results.append(metrics)
    except Exception as e:
        print(f"评估样本 {idx+1} 时出错: {e}")
        errors += 1
        continue

print(f"评估完成，{errors} 个样本评估失败")

# 7. 计算平均指标
if results:
    avg_metrics = {key: sum(item[key] for item in results if key in item) / len(results) 
                  for key in results[0].keys()}
    
    print("\n测试结果:")
    for metric, value in avg_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
else:
    print("没有有效的评估结果")

# 8. 保存详细结果
detailed_results = pd.DataFrame(results)
detailed_results.to_csv("test_results.csv")
print("\n详细结果已保存至test_results.csv")

# 9. 保存样本级别结果
samples_df = pd.DataFrame({
    "sample_id": list(range(len(test_dataset))),
    "predicted_ipa": [item["predicted_ipa"] for item in test_dataset],
    "gold_ipa": [item["ipa"] for item in test_dataset]
})
samples_df.to_csv("test_samples.csv", index=False)
print("样本预测结果已保存至test_samples.csv")
