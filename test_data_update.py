import pandas as pd
import os

def update_test_data():
    """向test_data.csv添加Chinese列并填入中文示例"""
    try:
        df = pd.read_csv('test_data.csv')
        print(f"成功读取测试数据，共{len(df)}行")
    except Exception as e:
        print(f"读取测试数据时出错：{e}")
        return
    
    # 添加Chinese列（如果不存在）
    if 'Chinese' not in df.columns:
        df['Chinese'] = None
        print("已添加Chinese列")
    
    # 为部分空白ID添加中文示例
    chinese_examples = [
        "你好，世界",
        "我爱中国",
        "语音识别技术",
        "北京是中国的首都",
        "中文转音标测试",
        "机器学习很有趣",
        "自然语言处理",
        "深度学习模型",
        "人工智能与大数据",
        "计算机视觉研究"
    ]
    
    # 查找未完成且无IPA记录的行
    empty_rows = df[(df['Done'] == 0) & (df['Poor quality'] == 0) & pd.isna(df['IPA'])].index
    print(f"找到{len(empty_rows)}个可填充的空行")
    
    # 确保不超出示例数量
    fill_count = min(len(empty_rows), len(chinese_examples))
    
    # 填充中文示例
    for i in range(fill_count):
        row_idx = empty_rows[i]
        df.at[row_idx, 'Chinese'] = chinese_examples[i]
        print(f"行 {row_idx}：添加中文示例 '{chinese_examples[i]}'")
    
    # 保存更新后的CSV
    df.to_csv('test_data_with_chinese.csv', index=False)
    print("已将更新后的数据保存到 test_data_with_chinese.csv")

if __name__ == "__main__":
    update_test_data() 