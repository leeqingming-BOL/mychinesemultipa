# 中文MultIPA扩展

这个项目是MultIPA的扩展版本，添加了对中文（普通话）的支持。MultIPA是一个将语音转录为国际音标(IPA)的自动模型。

## 项目修改内容

1. 创建了`converter/chinese_to_ipa.py`模块，实现了中文到IPA的转换
2. 在`preprocess.py`中添加了对中文的处理支持
3. 在`data_utils.py`中添加了中文处理函数和Preprocessor类的中文方法
4. 在`main.py`中添加了中文数据加载支持
5. 创建了测试脚本`test_chinese_ipa.py`用于验证中文到IPA的转换
6. 创建了数据处理脚本`process_test_data.py`和`test_data_update.py`，用于处理中文测试数据

## 中文到IPA的转换

中文到IPA的转换使用了以下步骤：

1. 使用`pypinyin`库将汉字转换为拼音
2. 将拼音映射到对应的IPA音标
3. 处理声调和特殊发音规则
4. 连接IPA序列，形成完整的音标表示

## 如何使用

### 环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 测试中文到IPA转换

```bash
python test_chinese_ipa.py
```

### 处理中文测试数据

```bash
# 更新测试数据，添加中文示例
python test_data_update.py

# 处理中文数据并转换为IPA
python process_test_data.py
```

### 训练模型

```bash
# 预处理中文数据
python preprocess.py -l zh --cache_dir ./cache --clear_cache

# 训练包含中文的多语言模型
python main.py -l ja pl mt hu fi el ta en zh -tr 1000 1000 1000 1000 1000 1000 1000 1000 1000 -te 200 200 200 200 200 200 200 200 200 -v vocab.json -e 10
```

## 中文IPA映射示例

| 中文 | 拼音 | IPA |
|------|------|-----|
| 你好 | ni3 hao3 | ni˨˩˦ xau̯˨˩˦ |
| 中国 | zhong1 guo2 | ʈʂʊŋ˥ ku̯o˧˥ |
| 北京 | bei3 jing1 | pei̯˨˩˦ tɕiŋ˥ |
| 语音 | yu3 yin1 | y˨˩˦ yin˥ |

## 注意事项

- 中文转IPA的规则基于普通话标准发音
- 当前实现包含声调标记，可根据需要在转换中忽略
- 特殊拼音组合有特殊的IPA映射规则

## 联系方式

如果您对该中文扩展有任何问题或建议，请通过Issues联系我们。 