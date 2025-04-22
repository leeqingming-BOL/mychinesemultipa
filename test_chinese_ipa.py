from converter.chinese_to_ipa import Chinese2IPA

def test_chinese_to_ipa():
    """测试中文到国际音标的转换功能"""
    test_sentences = [
        "你好，世界",
        "我爱中国",
        "语音识别技术",
        "北京是中国的首都"
    ]
    
    converter = Chinese2IPA()
    
    print("中文转IPA测试：")
    print("=" * 50)
    
    for sentence in test_sentences:
        print(f"原文：{sentence}")
        ipa = converter.convert_sentence_to_ipa(sentence)
        print(f"IPA：{ipa}")
        print("-" * 50)

if __name__ == "__main__":
    test_chinese_to_ipa() 