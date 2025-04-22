import re
import pypinyin

class Chinese2IPA:
    # 拼音到IPA的映射
    pinyin_to_ipa = {
        # 声母
        "b": "p",
        "p": "pʰ",
        "m": "m",
        "f": "f",
        "d": "t",
        "t": "tʰ",
        "n": "n",
        "l": "l",
        "g": "k",
        "k": "kʰ",
        "h": "x",
        "j": "tɕ",
        "q": "tɕʰ",
        "x": "ɕ",
        "zh": "ʈʂ",
        "ch": "ʈʂʰ",
        "sh": "ʂ",
        "r": "ʐ",
        "z": "ts",
        "c": "tsʰ",
        "s": "s",
        "w": "w",
        "y": "j",
        
        # 韵母，更新为更准确的IPA表示
        "a": "a",
        "o": "o",
        "e": "ɤ",
        "i": "i",
        "u": "u",
        "ü": "y",
        "ai": "aɪ",
        "ei": "eɪ",
        "ao": "ɑʊ",
        "ou": "oʊ",
        "ia": "ia",
        "ie": "iɛ",
        "iao": "iɑʊ",
        "iu": "ioʊ",
        "ua": "ua",
        "uo": "uɔ",
        "uai": "uaɪ",
        "ui": "ueɪ",
        "üe": "yɛ",
        "er": "ɚ",
        "an": "an",
        "en": "ən",
        "in": "in",
        "ian": "iɛn",
        "uan": "uan",
        "üan": "yɛn",
        "ün": "yn",
        "ang": "aŋ",
        "eng": "əŋ",
        "ing": "iŋ",
        "iang": "iaŋ",
        "uang": "uaŋ",
        "ong": "ʊŋ",
        "iong": "iʊŋ",
        
        # 声调标记，更新为标准IPA声调符号
        "1": "˥", # 第一声（阴平）高平调
        "2": "˧˥", # 第二声（阳平）中升调
        "3": "˨˩˦", # 第三声（上声）降升调
        "4": "˥˩", # 第四声（去声）降调
        "5": "", # 轻声，不标调
    }
    
    # 特殊发音规则处理，更新为更准确的表示
    special_rules = {
        "zi": "tsɨ",
        "ci": "tsʰɨ",
        "si": "sɨ",
        "zhi": "ʈʂɨ",
        "chi": "ʈʂʰɨ",
        "shi": "ʂɨ",
        "ri": "ʐɨ",
        "yi": "i",
        "wu": "u",
        "yu": "y",
        "yue": "yɛ",
        "ye": "iɛ",
        "ying": "iŋ",
    }
    
    def __init__(self):
        pass
    
    def remove_chinese_punct(self, sent: str) -> str:
        """
        移除中文标点符号
        """
        punctuation = r'[，。！？：；''""、【】《》（）]'
        sent = re.sub(punctuation, "", sent).lower()
        return sent
    
    def convert_pinyin_to_ipa(self, pinyin: str) -> str:
        """
        将单个拼音转换为IPA
        """
        # 处理声调
        tone = ""
        if pinyin[-1].isdigit():
            tone = pinyin[-1]
            pinyin = pinyin[:-1]
        
        # 检查特殊规则
        if pinyin in self.special_rules:
            ipa = self.special_rules[pinyin]
        else:
            # 分离声母和韵母
            initial = ""
            final = pinyin
            
            # 处理声母
            for i in range(min(2, len(pinyin)), 0, -1):
                if pinyin[:i] in self.pinyin_to_ipa:
                    initial = pinyin[:i]
                    final = pinyin[i:]
                    break
            
            # 转换为IPA
            ipa = ""
            if initial:
                ipa += self.pinyin_to_ipa.get(initial, initial)
            if final:
                ipa += self.pinyin_to_ipa.get(final, final)
        
        # 添加声调
        if tone and tone in self.pinyin_to_ipa:
            ipa += self.pinyin_to_ipa[tone]
            
        return ipa
    
    def convert_sentence_to_ipa(self, sent: str) -> str:
        """
        将整个中文句子转换为IPA
        """
        # 移除标点
        sent = self.remove_chinese_punct(sent)
        
        # 使用pypinyin转换为拼音
        pinyin_list = pypinyin.lazy_pinyin(sent, style=pypinyin.Style.TONE3)
        
        # 转换每个拼音为IPA
        ipa_list = [self.convert_pinyin_to_ipa(py) for py in pinyin_list]
        
        # 连接IPA序列，用空格分隔
        return " ".join(ipa_list)
    
    @classmethod
    def chinese_generate_ipa(cls, sent: str) -> str:
        """
        类方法，直接调用转换
        """
        converter = cls()
        return converter.convert_sentence_to_ipa(sent) 