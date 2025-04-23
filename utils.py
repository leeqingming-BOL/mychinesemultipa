import numpy as np
import pandas as pd
from transformers import Wav2Vec2CTCTokenizer

# df = pd.read_csv("features.csv", index_col=0)

def convert_features_to_numeric(feature_values):
    """将特征值（'+', '-', '0'等）转换为数值表示
    
    Args:
        feature_values: 包含特征值的数组或列表
        
    Returns:
        转换后的数值数组
    """
    numeric_features = []
    for val in feature_values:
        if val == '+':
            numeric_features.append(1.0)
        elif val == '-':
            numeric_features.append(-1.0)
        else:
            numeric_features.append(float(val))
    return np.array(numeric_features)

def cos_sim(v1, v2):
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0:
        denominator = 0.001
    return np.dot(v1, v2) / denominator

def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]

def LPhD(token1, token2, df):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            # penalty mitigation
            raw_t1_f = df.loc[token1[t1-1], :].to_numpy()[1:]
            raw_t2_f = df.loc[token2[t2-1], :].to_numpy()[1:]
            
            # 使用辅助函数转换特征值
            t1_f = convert_features_to_numeric(raw_t1_f)
            t2_f = convert_features_to_numeric(raw_t2_f)
            
            penalty = 1 - cos_sim(t1_f, t2_f)

            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + penalty
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + penalty
                else:
                    distances[t1][t2] = c + penalty
    
    return distances[len(token1)][len(token2)]

# Spacing Modifier Letters
sml = set()
for i in range(int(0x2b0), int(0x36f)+1):
    sml.add(chr(i))

def retokenize_ipa(sent: str):
    tie_flag = False
    modified = []
    for i in range(len(sent)):
        if tie_flag:
            tie_flag = False
            continue
        if sent[i] in sml:
            if i == 0:
                # when the space modifier letter comes at the index 0
                modified.append(sent[i])
                continue
            modified[-1] += sent[i]
            if sent[i] == "\u0361":
                # tie bar
                modified[-1] += sent[i+1]
                tie_flag = True
        else:
            modified.append(sent[i])
    return modified

def combine_features(phone: str, df):
    # global phone_not_found
    features = np.array([0] * (df.shape[1] - 1))
    for p in phone:
        if p not in set(df.index):
            print("The IPA {} (U+{}) not found in the feature table. We will use zeroed out feature vector instead.".format(p, hex(ord(p))))
            f = np.array([0] * (df.shape[1] - 1))
            # add the unknown phone and its unicode to the dict so that at the end of the evaluation
            # we can get the list of phones unsupported in the feature table
            # phone_not_found[p] = hex(ord(p))
        else:
            # 获取特征并转换为数值类型
            raw_features = df.loc[p, :].to_numpy()[1:]
            f = convert_features_to_numeric(raw_features)
        # print(f)
        features = np.add(features, f)
        # ReLU if necessary
    return features

def preprocessing_combine(sent: str, df) -> pd.DataFrame:
    # df is the feature table
    sent_index = retokenize_ipa(sent)
    sent_array = [[0 for i in range(1, df.shape[1])] for j in range(len(sent_index))]
    sent_df = pd.DataFrame(sent_array, index=sent_index, columns=df.columns[1:])
    # print(sent_df)
    for i, phone in enumerate(sent_df.index):
        if phone in df.index:
            # 获取特征并转换为数值类型
            raw_features = df.loc[phone].to_numpy()[1:]
            numeric_features = convert_features_to_numeric(raw_features)
            sent_df.iloc[i] = numeric_features
        else:
            features = combine_features(phone, df)
            sent_df.iloc[i] = features
        # print(phone, features)
    return sent_df

def LPhD_combined(df1, df2):
    distances = np.zeros((df1.shape[0] + 1, df2.shape[0] + 1))

    for t1 in range(df1.shape[0] + 1):
        distances[t1][0] = t1
    for t2 in range(df2.shape[0] + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    for t1 in range(1, df1.shape[0] + 1):
        for t2 in range(1, df2.shape[0] + 1):
            # penalty mitigation
            t1_f = df1.iloc[t1-1].values  # 使用values确保我们获取到numpy数组
            t2_f = df2.iloc[t2-1].values
            penalty = 1 - cos_sim(t1_f, t2_f)

            # 我们直接比较两个行的值是否相等，不使用to_numpy()[1:]
            # 因为df1和df2已经是预处理后的数据框
            if np.array_equal(t1_f, t2_f):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + penalty
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + penalty
                else:
                    distances[t1][t2] = c + penalty
    
    return distances[df1.shape[0]][df2.shape[0]]

def phoneme_error_rate(df1, df2):
    # df2 should be the reference
    df1_list = df1.index
    df2_list = df2.index
    for i, c in enumerate(df1_list):
        if pd.isna(c):
            df1_list.pop(i)
    for i, c in enumerate(df2_list):
        if pd.isna(c):
            df2_list.pop(i)
    phone_LD = levenshteinDistanceDP(df1_list, df2_list)
    ref_length = len(df2_list)
    return phone_LD / ref_length

def compute_all_metrics(pred: str, gold: str, df) -> dict:
    pred = pred.replace("g", "ɡ") # different unicode characters!
    gold = gold.replace("g", "ɡ")
    
    # Levenshtein distance
    ld = levenshteinDistanceDP(pred, gold)

    # Character Error Rate
    cer = ld / len(gold) if len(gold) > 0 else 0

    # Phoneme Error Rate
    df_pred = preprocessing_combine(pred, df)
    df_gold = preprocessing_combine(gold, df)
    per = phoneme_error_rate(df_pred, df_gold)

    # Levenshtein Phone Distance
    lphd = LPhD_combined(df_pred, df_gold)

    # Feature-weighted Phone Error Rate based on LPhD
    fper = lphd / df_gold.shape[0] if df_gold.shape[0] > 0 else 0
    # shape[0] gives the length of the gold transcription

    output = {"Levenshtein Distance": ld,
              "Character Error Rate": cer,
              "Phoneme Error Rate": per,
              "Levenshtein Phone Distance": lphd,
              "Feature-weighted Phone Error Rate": fper}

    return output

def compute_only_fper(pred: str, gold: str, df) -> int:
    """Compute Feature-weighted Phone Error Rate.
    
    Args:
        pred: Predicted IPAs
        gold: Gold (label) IPAs
        df: IPA table with features
    Returns:
        fper: Feature-weighted Phone Error Rate for pred
    """
    pred = pred.replace("g", "ɡ") # different unicode characters!
    gold = gold.replace("g", "ɡ")
    df_pred = preprocessing_combine(pred, df)
    df_gold = preprocessing_combine(gold, df)

    # Levenshtein Phone Distance
    lphd = LPhD_combined(df_pred, df_gold)

    # Feature-weighted Phone Error Rate based on LPhD
    fper = lphd / df_gold.shape[0] * 100
    
    return fper
