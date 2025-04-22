import pandas as pd
import os
from converter.chinese_to_ipa import Chinese2IPA

def process_test_data():
    """处理test_data_with_chinese.csv中的中文测试数据并转换为IPA"""
    # 读取测试数据
    try:
        df = pd.read_csv('test_data_with_chinese.csv')
        print(f"成功读取测试数据，共{len(df)}行")
    except Exception as e:
        print(f"读取测试数据时出错：{e}")
        return
    
    # 创建中文转IPA转换器
    converter = Chinese2IPA()
    
    # 创建结果目录
    output_dir = 'chinese_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 为每个尚未处理的测试样本生成IPA转写
    results = []
    
    for index, row in df.iterrows():
        # 检查是否已经有IPA转写以及中文文本
        chinese_text = row.get('Chinese', None)
        if pd.isna(row['IPA']) and row['Done'] == 0 and row['Poor quality'] == 0 and not pd.isna(chinese_text):
            print(f"处理第{index}行数据")
            
            # 转换为IPA
            ipa = converter.convert_sentence_to_ipa(chinese_text)
            
            # 更新结果
            results.append({
                'ID': row['ID'],
                'Chinese': chinese_text,
                'IPA': ipa,
                'Done': 1
            })
            
            print(f"  原文：{chinese_text}")
            print(f"  IPA：{ipa}")
    
    # 将结果保存到文件
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(f'{output_dir}/processed_results.csv', index=False)
        print(f"处理了{len(results)}个样本，结果已保存到{output_dir}/processed_results.csv")
        
        # 更新原始CSV
        for result in results:
            idx = df[df['ID'] == result['ID']].index[0]
            df.at[idx, 'IPA'] = result['IPA']
            df.at[idx, 'Done'] = 1
        
        df.to_csv('test_data_with_chinese_updated.csv', index=False)
        print(f"已更新原始数据并保存到test_data_with_chinese_updated.csv")
    else:
        print("没有需要处理的样本")

if __name__ == "__main__":
    process_test_data() 