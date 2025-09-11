import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class StockPredictionTransformer(nn.Module):
    def __init__(self, etf_embedding_dim, etf_feature_dim, macro_feature_dim, input_dim, transformer_dim, seq_len, output_dim,
                 num_etfs, num_heads, num_layers):
        super().__init__()
        self.seq_len = seq_len
        # 輸入嵌入層 (Input Embedding)
        self.elu = nn.ELU()
        self.etf_feature_embedding = nn.Linear(etf_feature_dim, input_dim)
        self.macro_embedding = nn.Linear(macro_feature_dim, input_dim)

        # Transformer Encoder 層
        encoder_layers = nn.TransformerEncoderLayer(d_model=2 * input_dim, # *2 for etf and macro output dim
                                                    nhead=num_heads,
                                                    dim_feedforward=transformer_dim) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # ----- 加入 ETF 嵌入層 -----
        # 為每個 ETF 學習一個獨特的向量
        self.etf_embedding = nn.Embedding(num_etfs, etf_embedding_dim)

        # ----- 加入位置編碼 -----
        # 這裡使用標準的正弦位置編碼
        self.positional_encoding = self._generate_positional_encoding(seq_len, 2 * input_dim)
        self.register_buffer('pe', self.positional_encoding) # 將位置編碼註冊為 buffer，它不會被視為模型參數，但在 state_dict 中會被保存和加載

        # Attention pooling 層
        self.attention_score_layer = nn.Linear(2 * input_dim, 1)

        # Droupout 層
        self.pooling_dropout = nn.Dropout(0.3)
        # self.output_dropout = nn.Dropout(0.1)

        # 融合層 (簡單拼接後的全連接層)
        self.fusion_layer = nn.Linear(2 * input_dim + etf_embedding_dim, seq_len) # 拼接 ETF 和 Macro 特徵

        # 輸出層 (回歸預測下週漲跌幅度)
        self.output_layer = nn.Linear(seq_len, output_dim)

    # 生成標準的正弦位置編碼
    def _generate_positional_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 需要將形狀調整為 (seq_len, 1, d_model) 以便於加到 (seq_len, batch_size, d_model) 的輸入上 (通過廣播)
        pe = pe.unsqueeze(1)
        return pe

    def forward(self, etf_features, macro_features, etf_indices):
        # 輸入嵌入
        etf_embedded = self.elu(self.etf_feature_embedding(etf_features)) # [batch_size, seq_len, input_dim] - 假設修正後形狀為 3 維
        macro_embedded = self.elu(self.macro_embedding(macro_features)) # [batch_size, seq_len, input_dim] - 假設修正後形狀為 3 維

        # Transformer Encoder (需要調整輸入形狀為 (seq_len, batch_size, feature_dim))
        # 這裡假設 batch_first=False，所以需要將 batch_size 維度放到第二維
        etf_embedded = etf_embedded.transpose(0, 1)
        macro_embedded = macro_embedded.transpose(0, 1)

        # 拼接 ETF 和 Macro 特徵 (在 feature 維度拼接, dim=2)
        fused_features = torch.cat((etf_embedded, macro_embedded), dim=2) # [seq_len, batch_size, 2*input_dim]

        # ----- 加入位置編碼 -----
        # 將位置編碼加到 Transformer 的輸入上
        # self.pe 是 [seq_len, 1, d_model]，加到 [seq_len, batch_size, d_model] 會通過廣播
        # 注意：位置編碼的序列長度要與當前輸入的序列長度匹配
        # 如果訓練和測試的 seq_len 不同，這裡需要調整或重新生成位置編碼
        fused_features = fused_features + self.pe[:self.seq_len, :].to(fused_features.device)

        # 透過 Transformer Encoder
        transformer_output = self.transformer_encoder(fused_features) # [seq_len, batch_size, 2*input_dim]

        # ----- Learnable Attention Pooling -----
        # 1. 計算原始注意力分數
        # [seq_len, batch_size, input_dim] -> [seq_len, batch_size, 1]
        attention_scores = self.attention_score_layer(transformer_output)

        # 2. 歸一化注意力權重
        # 在序列長度維度 (dim=0) 上應用 Softmax
        # [seq_len, batch_size, 1] -> [seq_len, batch_size, 1]
        attention_weights = torch.softmax(attention_scores, dim=0)

        # 3. 使用 torch.bmm 計算加權和
        # 需要將張量形狀調整為 (batch_size, n, m) 和 (batch_size, m, p)
        # attention_weights: [seq_len, batch_size, 1] -> permute(1, 2, 0) -> [batch_size, 1, seq_len] (αᵀ 形態)
        # transformer_output: [seq_len, batch_size, input_dim] -> permute(1, 0, 2) -> [batch_size, seq_len, input_dim] (H 形態)
        # torch.bmm([batch_size, 1, seq_len], [batch_size, seq_len, input_dim]) -> [batch_size, 1, input_dim]
        # 將序列長度維度 (0) 和 batch_size 維度 (1) 交換
        attention_weights_permuted = attention_weights.permute(1, 2, 0) # [batch_size, 1, seq_len]
        transformer_output_permuted = transformer_output.permute(1, 0, 2) # [batch_size, seq_len, input_dim]

        # 執行批量矩陣乘法
        pooled_output = torch.bmm(attention_weights_permuted, transformer_output_permuted) # [batch_size, 1, input_dim]

        # 移除中間的 1 維度，得到 [batch_size, input_dim] 的形狀
        pooled_output = pooled_output.squeeze(1) # [batch_size, input_dim]

        # Dropout regularization
        pooled_output = self.pooling_dropout(pooled_output)

        # ----- 獲取 ETF 嵌入向量並拼接 -----
        # etf_indices 已經是 [batch_size]
        etf_embedded_vector = self.etf_embedding(etf_indices) # [batch_size, etf_embedding_dim]

        # 將 Attention Pooling 的輸出與 ETF 嵌入向量拼接
        fusion_input = torch.cat((pooled_output, etf_embedded_vector), dim=1) # [batch_size, 2 * input_dim + etf_embedding_dim]

        # 融合層
        fused_output = self.elu(self.fusion_layer(fusion_input)) # [batch_size, seq_len]
        # fused_output = self.output_dropout(fused_output)

        # 輸出層
        prediction = self.output_layer(fused_output) # [batch_size, output_dim]

        return prediction
    
class StockDataset(Dataset):
    def __init__(self, etf_list:List, etf_df:pd.DataFrame, macro_df:pd.DataFrame, test_percentage:float = 0.2,
                 sequence_length:int = 12, sampling_interval:int = 2):
        self.etf_list = etf_list
        self.etf_df = etf_df
        self.macro_df = macro_df
        self.sequence_length = sequence_length # 設定序列長度 (例如 12 周) - 可以根據需要調整
        self.sampling_interval = sampling_interval
        self.etf_sample_end_idx = {}
        self.num_etf_features = 0
        self.num_macros = len(macro_df.columns)

        # 資料預處理和特徵工程
        self.etf_symbol_to_index = {symbol: i for i, symbol in enumerate(etf_list)}
        self.etf_feature_mean_std = {}
        self.processed_data = self._preprocess_data()
        num_test_samples_per_etf = int(len(self.processed_data) * test_percentage) // len(self.etf_list)
        self.test_idx = np.array([np.arange(num_test_samples_per_etf) + 1 + end_idx - num_test_samples_per_etf for end_idx in self.etf_sample_end_idx.values()]).flatten()
        self.train_idx = np.setdiff1d(np.arange(len(self.processed_data)), self.test_idx)
        self.train_test_flag = 0 # 0 for train, 1 for test

    def _preprocess_data(self):
        num_etfs = len(self.etf_list)
        columns = self.etf_df.columns
        self.num_etf_features = (len(columns) - 1) // num_etfs  # Minus 1 for Date
        
        processed_list = [] # 樣本列表
        dates_full = self.etf_df['Date'].values
        macro_features_full = self.macro_df.values
        scaler = StandardScaler()

        for i in range(1, len(columns), self.num_etf_features):
            etf_features_full = self.etf_df[columns[i:i + self.num_etf_features]]
            etf_symbol = etf_features_full.columns.values[0].split("_")[0]
            targets_full = etf_features_full[f'{etf_symbol}_price_change'].values.reshape(-1, 1)
            etf_index = self.etf_symbol_to_index[etf_symbol] # 獲取該 ETF 的索引
 
            start_indexes = []
            valid_indexes = set() # 紀錄可用於生成樣本的索引，將用於標準化樣本
            for j in range(self.sequence_length, len(etf_features_full), self.sampling_interval): # 滑動窗口尋找可生成樣本開始位置
                start_index = j - self.sequence_length # 窗口開始索引
                end_index = j # 窗口結束索引 + 1 (即不包含)

                if etf_features_full.iloc[start_index:end_index].isna().values.any() or \
                   np.isnan(macro_features_full[start_index:end_index]).any() or \
                   np.isnan(targets_full[end_index]): # Skip samples with NaN data
                    continue

                start_indexes.append(start_index)
                if start_index not in valid_indexes:
                    valid_indexes.update(np.arange(self.sequence_length) + start_index)

            scaler.fit(etf_features_full.iloc[list(valid_indexes)].values)
            self.etf_feature_mean_std[etf_symbol] = {'mean':scaler.mean_, 'scale':scaler.scale_}

            for start_index in start_indexes:
                end_index = start_index + self.sequence_length # 窗口結束索引 + 1 (即不包含)
                etf_features = scaler.transform(etf_features_full.values[start_index:end_index]) # 取過去 seq_len 單位(周)的 ETF 特徵，並做全局標準化
                target = targets_full[end_index] # 取當單位(周) (end_index) 的目標值
                dates = ",".join(dates_full[start_index:end_index]) # 取過去 seq_len 單位(周)的日期
                macro_features = macro_features_full[start_index:end_index] # 取過去 seq_len 單位(周)的 Macro 特徵

                processed_list.append({ # 生成單個樣本
                    'etf_features': torch.tensor(etf_features, dtype=torch.float32),
                    'macro_features': torch.tensor(macro_features, dtype=torch.float32),
                    'targets': torch.tensor(target, dtype=torch.float32), # 目標值現在是單個數值
                    'dates': dates,
                    'etf_symbol': etf_symbol,
                    'etf_index': etf_index # 新增：儲存 ETF 索引
                })

            self.etf_sample_end_idx[etf_symbol] = len(processed_list) - 1
        
        print(f"Generated {len(processed_list)} samples.")
                
        return processed_list # 返回樣本列表
    
    def train(self):
        self.train_test_flag = 0

    def test(self):
        self.train_test_flag = 1

    def __len__(self):
        return len(self.train_idx) if self.train_test_flag == 0 else len(self.test_idx)

    def __getitem__(self, idx):
        return self.processed_data[self.train_idx[idx]] if self.train_test_flag == 0 else self.processed_data[self.test_idx[idx]]
