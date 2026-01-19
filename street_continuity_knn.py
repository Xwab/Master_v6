#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Added for attention

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== 0) 配置 =====================
# 请根据实际情况修改路径
excel_path = "/home/users/yyb/last/567append_all.xlsx"
sheet_name = "Sheet1"

# 邻接矩阵路径 (已确认存在)
adj_paths = {
    2000: "/workspace/W2000_raw.txt",
    2010: "/workspace/W2010_raw.txt",
    2020: "/workspace/W2020_full_raw.txt"
}

# === 关键配置 ===
# Excel中对应街道代码的列名 (必须存在，用于和W文件匹配)
# 如果您的Excel里代码列不是"code"，请在这里修改
CODE_COL_NAME = "code" 

USE_KNN_FEATURES = True      # KNN特征 (作为辅助环境背景)
USE_ADJ_FEATURES = True      # 是否使用邻接矩阵
MAX_ADJ_NEIGHBORS = 6        # 每个样本最多保留几个物理邻居进入注意力网络 (不足补0)

out_cv_xlsx   = "/home/users/yyb/last/cv_residual_nn_knn_separated.xlsx"
out_pred_xlsx = "/home/users/yyb/last/completed_residual_nn_knn_separated.xlsx"

# 创建输出目录
for p in [out_cv_xlsx, out_pred_xlsx]:
    try:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory for {p}: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ===================== 1) 读入数据 =====================
na_tokens = ["", " ", "  ", "\t", "\n", "NA", "N/A", "na", "n/a", ".", "None", "null", "NULL"]
try:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, na_values=na_tokens, keep_default_na=True)
except FileNotFoundError:
    print(f"Error: File not found at {excel_path}. Please check the path.")
    # 生成假数据用于演示
    print("Generating dummy data for testing logic...")
    data = []
    names = [f"Street_{i}" for i in range(100)]
    for y in [2000, 2010, 2020]:
        for i, n in enumerate(names):
            # 模拟代码: 简单起见假设与行号相关
            # 模拟部分缺失: 2020年部分target缺失
            is_target_missing = (y == 2020) and (int(n.split('_')[1]) > 70)
            row = {"名称": n, "year": y, "area": 100, CODE_COL_NAME: (y//1000)*1000 + i} 
            for v in ["age65sacle", "malesacle", "nong_incomesacle", "shengchan_incomesacle", 
                      "qita_incomesacle", "unemploysacle", "tizhisacle", "outsidesacle"]:
                row[v] = np.random.rand()
            for t in ["bungalowsacle", "nowcsacle", "nobathroomsacle", "nokitchensacle"]:
                row[t] = np.nan if is_target_missing else np.random.rand()
            for t in ["bungalow", "nowc", "nobathroom", "nokitchen"]:
                row[t] = np.nan
            data.append(row)
    df = pd.DataFrame(data)

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["名称"] = df["名称"].astype(str).str.strip()

# 确保代码列存在
if CODE_COL_NAME not in df.columns:
    print(f"⚠️ Warning: Column '{CODE_COL_NAME}' not found. Using Index as dummy code.")
    df[CODE_COL_NAME] = df.index
else:
    try:
        df[CODE_COL_NAME] = pd.to_numeric(df[CODE_COL_NAME], errors='raise').astype("Int64")
    except:
        df[CODE_COL_NAME] = df[CODE_COL_NAME].astype(str).str.strip()

# 定义变量
target_base = ["bungalowsacle", "nowcsacle", "nobathroomsacle", "nokitchensacle"]
target_base_no_sacle = ["bungalow", "nowc", "nobathroom", "nokitchen"]

vars_all = [
    "area", "age65sacle", "malesacle",
    "nong_incomesacle", "shengchan_incomesacle", "qita_incomesacle",
    "unemploysacle", "tizhisacle", "outsidesacle",
    "bungalowsacle", "nowcsacle", "nobathroomsacle", "nokitchensacle"
]
feature_base = [v for v in vars_all if v not in target_base]

cols_to_numeric = vars_all + target_base_no_sacle
for c in cols_to_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Wide转换 (保留 Code 列)
pivot_values = cols_to_numeric + [CODE_COL_NAME]
pivot_values = [c for c in pivot_values if c in df.columns]
wide = df.pivot_table(index="名称", columns="year", values=pivot_values, aggfunc="first")
wide.columns = [f"{v}_{int(y)}" for (v, y) in wide.columns]
wide = wide.reset_index()

print("Data Loaded. Wide shape:", wide.shape)

# ===================== 1.5) 加载邻接矩阵 =====================
def load_adj_matrix(file_path):
    """
    解析邻接矩阵txt文件。
    假设格式: 第一行是数量 N。后续 N 行，每行第一个数是 Code，后面 N 个 0/1 表示邻接关系。
    返回: {code: [neighbor_code_1, neighbor_code_2, ...]}
    """
    adj_map = {}
    path_obj = Path(file_path)
    if not path_obj.exists():
        print(f"⚠️ Adjacency file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        if not lines: return {}
        
        # 假设第一行是数量，跳过
        all_codes = []
        matrix_rows = []
        
        # 从第二行开始读取
        for l in lines[1:]:
            parts = l.split()
            code = parts[0]
            try:
                code_val = int(float(code)) # handle '1001.0'
            except:
                code_val = code
            all_codes.append(code_val)
            # 解析 0/1 向量
            matrix_rows.append([int(float(p)) for p in parts[1:]])
            
        # 构建邻接表
        for i, code in enumerate(all_codes):
            neighbors = []
            if i < len(matrix_rows):
                row_vec = matrix_rows[i]
                length = min(len(row_vec), len(all_codes))
                for j in range(length):
                    if row_vec[j] == 1 and i != j: # 排除自环
                        neighbors.append(all_codes[j])
            adj_map[code] = neighbors
            
        print(f"Loaded Adj Matrix {path_obj.name}: {len(adj_map)} locations.")
        return adj_map
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

adj_data = {}
if USE_ADJ_FEATURES:
    print("Loading Adjacency Matrices...")
    for year, path in adj_paths.items():
        adj_data[year] = load_adj_matrix(path)

# ===================== 2) 筛选连续存在的街道/乡镇 =====================
def get_valid_names(df_wide, y1, y2, check_cols_prefix):
    valid_indices = []
    for idx, row in df_wide.iterrows():
        # 必须所有特征在 y1 和 y2 都非空
        has_y1 = all(pd.notna(row.get(f"{v}_{y1}", np.nan)) for v in feature_base)
        has_y2 = all(pd.notna(row.get(f"{v}_{y2}", np.nan)) for v in feature_base)
        
        if has_y1 and has_y2:
            valid_indices.append(row["名称"])
    return set(valid_indices)

names_00_10 = get_valid_names(wide, 2000, 2010, feature_base)
names_10_20 = get_valid_names(wide, 2010, 2020, feature_base)

print(f"Valid pairs 2000-2010: {len(names_00_10)}")
print(f"Valid pairs 2010-2020: {len(names_10_20)}")

# ===================== 辅助：构建快速特征查找表 (全量) =====================
print("Building Lookup Maps...")
lookup_maps = {} 
all_years = [2000, 2010, 2020]

for y in all_years:
    lookup_maps[y] = {}
    col_code = f"{CODE_COL_NAME}_{y}"
    relevant_cols = [f"{v}_{y}" for v in vars_all]
    
    if col_code in wide.columns:
        valid_rows = wide.dropna(subset=[col_code])
        # 转成 dict: code -> list_of_values (按 vars_all 顺序)
        # 注意: 这里我们存储 dense array list，方便后续处理
        # 缺失值补0，因为是 lookup 给邻居用的
        for idx, row in valid_rows.iterrows():
            code = row[col_code]
            vals = []
            for col in relevant_cols:
                val = row.get(col, 0.0)
                if pd.isna(val): val = 0.0
                vals.append(val)
            lookup_maps[y][code] = vals

# ===================== 3) 核心处理函数：提取特征 + 邻居Tensor =====================
def process_period(wide_df, names_subset, year_prev, year_curr, is_training_data=True):
    sub_df = wide_df[wide_df["名称"].isin(names_subset)].copy()
    if sub_df.empty: return [], [], [], []
    sub_df = sub_df.reset_index(drop=True)
    
    # KNN 预计算 (保持不变)
    search_cols = [f"{v}_{year_prev}" for v in vars_all] + [f"{v}_{year_curr}" for v in feature_base]
    missing_cols = [c for c in search_cols if c not in sub_df.columns]
    for c in missing_cols: sub_df[c] = np.nan
    
    X_search = sub_df[search_cols].values
    imputer_knn = SimpleImputer(strategy="median")
    X_search_imputed = imputer_knn.fit_transform(X_search)
    scaler_knn = StandardScaler()
    X_search_scaled = scaler_knn.fit_transform(X_search_imputed)
    X_search_scaled = np.nan_to_num(X_search_scaled, nan=0.0)
    
    k = min(11, len(sub_df))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_search_scaled)
    _, indices = nbrs.kneighbors(X_search_scaled)
    
    # 准备循环
    X_samples, Y_residuals, Base_values, Meta_info = [], [], [], []
    col_idx_map = {c: i for i, c in enumerate(sub_df.columns)}
    data_matrix = sub_df.values
    
    adj_prev = adj_data.get(year_prev, {}) if USE_ADJ_FEATURES else {}
    lookup_prev = lookup_maps.get(year_prev, {})
    col_code_prev = f"{CODE_COL_NAME}_{year_prev}"

    dim_vars_all = len(vars_all)

    for i in range(len(sub_df)):
        name = sub_df.iloc[i]["名称"]
        
        # --- A. KNN 均值特征 (保留) ---
        neighbor_idxs = indices[i, 1:] 
        knn_feats = []
        if USE_KNN_FEATURES:
            if len(neighbor_idxs) == 0: neighbor_idxs = [i]
            # KNN - Prev Mean
            for v in vars_all:
                col_name = f"{v}_{year_prev}"
                vals = data_matrix[neighbor_idxs, col_idx_map.get(col_name, 0)]
                knn_feats.append(np.nanmean(vals.astype(float)))
            # KNN - Curr Mean
            for v in feature_base:
                col_name = f"{v}_{year_curr}"
                vals = data_matrix[neighbor_idxs, col_idx_map.get(col_name, 0)]
                knn_feats.append(np.nanmean(vals.astype(float)))

        # --- B. Spatial (Adj) 可训练特征 ---
        # 我们这里只取 Prev Year 的邻居的全量特征 (因为有Label，信息量最大)
        # 将其展开为 [N1_f1, N1_f2, ..., N2_f1, N2_f2...]
        # 具体的 Attention 聚合交给模型
        
        adj_raw_feats = [] 
        if USE_ADJ_FEATURES:
            my_code_prev = sub_df.at[i, col_code_prev] if col_code_prev in sub_df.columns else None
            
            found_neighbors = []
            if my_code_prev in adj_prev:
                nbr_codes = adj_prev[my_code_prev]
                for n_code in nbr_codes:
                    if n_code in lookup_prev:
                        found_neighbors.append(lookup_prev[n_code]) # list of floats
            
            # Pad or Truncate
            # 目标: 凑齐 MAX_ADJ_NEIGHBORS 个邻居，每个邻居有 dim_vars_all 个特征
            count = 0
            for n_feat in found_neighbors:
                if count >= MAX_ADJ_NEIGHBORS: break
                adj_raw_feats.extend(n_feat) # extend list
                count += 1
            
            # 补 0
            remaining = MAX_ADJ_NEIGHBORS - count
            adj_raw_feats.extend([0.0] * (remaining * dim_vars_all))
            
        # --- C. 自身特征 ---
        self_feats = []
        prev_vals_dict = {}
        for v in vars_all:
            col_name = f"{v}_{year_prev}"
            val = sub_df.at[i, col_name] if col_name in sub_df.columns else np.nan
            self_feats.append(val)
            prev_vals_dict[v] = val
            
        curr_vals_dict = {}
        for v in feature_base:
            col_name = f"{v}_{year_curr}"
            val = sub_df.at[i, col_name] if col_name in sub_df.columns else np.nan
            self_feats.append(val)
            curr_vals_dict[v] = val
            
        for v in feature_base:
            p = prev_vals_dict.get(v, np.nan)
            c = curr_vals_dict.get(v, np.nan)
            if pd.notna(p) and pd.notna(c): self_feats.append(c - p)
            else: self_feats.append(np.nan)
        
        # 组合: [Self, KNN_Mean, Adj_Raw_Neighbors]
        full_row = self_feats + knn_feats + adj_raw_feats
        
        # --- Target ---
        base_vals = [prev_vals_dict.get(t, np.nan) for t in target_base]
        residuals, has_labels = [], True
        
        if is_training_data:
            for idx_t, t in enumerate(target_base):
                t_raw = target_base_no_sacle[idx_t]
                col_name_raw = f"{t_raw}_{year_curr}"
                val_raw = sub_df.at[i, col_name_raw] if col_name_raw in sub_df.columns else np.nan
                col_name = f"{t}_{year_curr}"
                val_curr = sub_df.at[i, col_name] if col_name in sub_df.columns else np.nan
                val_prev = base_vals[idx_t]
                
                if pd.notna(val_raw) and pd.notna(val_prev):
                    residuals.append(val_curr - val_prev)
                else:
                    residuals.append(np.nan)
                    has_labels = False

        X_samples.append(full_row)
        Y_residuals.append(residuals if is_training_data else [np.nan]*4)
        Base_values.append(base_vals)
        Meta_info.append({"名称": name, "Period": f"{year_prev}->{year_curr}", "HasLabel": has_labels})

    return X_samples, Y_residuals, Base_values, Meta_info

# ===================== 4) 构建数据集 =====================
print("Processing Period 1...")
X1, Y1, Base1, Meta1 = process_period(wide, names_00_10, 2000, 2010, True)
print("Processing Period 2...")
X2, Y2, Base2, Meta2 = process_period(wide, names_10_20, 2010, 2020, True)

X_train_list, Y_train_list, Base_train_list, Meta_train_list = [], [], [], []
X_pred_list, Base_pred_list, Meta_pred_list = [], [], []

for i, meta in enumerate(Meta1):
    if meta["HasLabel"] and not np.isnan(Y1[i]).any():
        X_train_list.append(X1[i])
        Y_train_list.append(Y1[i])
        Base_train_list.append(Base1[i])
        Meta_train_list.append(meta)

for i, meta in enumerate(Meta2):
    if meta["HasLabel"] and not np.isnan(Y2[i]).any():
        X_train_list.append(X2[i])
        Y_train_list.append(Y2[i])
        Base_train_list.append(Base2[i])
        Meta_train_list.append(meta)
    else:
        X_pred_list.append(X2[i])
        Base_pred_list.append(Base2[i])
        Meta_pred_list.append(meta)

X_train_raw = np.array(X_train_list, dtype=np.float32)
Y_train_res_raw = np.array(Y_train_list, dtype=np.float32)
X_pred_raw = np.array(X_pred_list, dtype=np.float32)
Pred_Base_Y = np.array(Base_pred_list, dtype=np.float32)

print(f"Train size: {X_train_raw.shape}, Pred size: {X_pred_raw.shape}")

# ===================== 5) 预处理 =====================
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train_raw)
X_pred_imputed = imputer.transform(X_pred_raw) if len(X_pred_raw) > 0 else X_pred_raw

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_pred_scaled = scaler.transform(X_pred_imputed) if len(X_pred_raw) > 0 else X_pred_raw

# 确定 Input 维度切分点
# Self + KNN 特征数量 (非 Adj 部分)
dim_vars_all = len(vars_all)
dim_feat_base = len(feature_base)
# Self features count: All_Prev + Base_Curr + Base_Delta
dim_self = dim_vars_all + dim_feat_base * 2 
# KNN features count: All_Prev_Mean + Base_Curr_Mean
dim_knn = (dim_vars_all + dim_feat_base) if USE_KNN_FEATURES else 0

static_dim = dim_self + dim_knn
adj_total_dim = (dim_vars_all * MAX_ADJ_NEIGHBORS) if USE_ADJ_FEATURES else 0

print(f"Feature dims -> Static: {static_dim}, Adj(Flattened): {adj_total_dim}, Total: {X_train_scaled.shape[1]}")

# ===================== 6) 可训练的 Attention 模型 =====================
class AttentionWeightedNet(nn.Module):
    def __init__(self, static_dim, neighbor_dim, max_neighbors):
        super(AttentionWeightedNet, self).__init__()
        self.static_dim = static_dim
        self.neighbor_dim = neighbor_dim # 单个邻居的特征维数 (vars_all)
        self.max_neighbors = max_neighbors
        self.use_adj = (max_neighbors > 0)
        
        # 1. 邻居特征编码器 (Neighbor Encoder)
        if self.use_adj:
            self.nbr_encoder = nn.Sequential(
                nn.Linear(neighbor_dim, 32),
                nn.ReLU(),
            )
            
            # 2. Attention Mechanism
            # Query 来自 Self (Static), Key 来自 Neighbor
            self.query_proj = nn.Linear(static_dim, 32)
            self.key_proj = nn.Linear(32, 32) # encoded nbr -> key
            
            # 聚合后的维度
            self.context_dim = 32
        else:
            self.context_dim = 0
            
        # 3. Main Prediction Network
        self.main_net = nn.Sequential(
            nn.Linear(static_dim + self.context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        # x shape: [batch, total_features]
        # Split input
        x_static = x[:, :self.static_dim]
        
        if self.use_adj:
            x_adj_flat = x[:, self.static_dim:]
            batch_size = x.shape[0]
            
            # Reshape: [Batch, Max_Neighbors, Neighbor_Dim]
            neighbors = x_adj_flat.view(batch_size, self.max_neighbors, self.neighbor_dim)
            
            # Encode neighbors: [Batch, Max_Neighbors, 32]
            nbr_encoded = self.nbr_encoder(neighbors)
            
            # Compute Attention Scores
            # Query: [Batch, 32] -> [Batch, 1, 32]
            query = self.query_proj(x_static).unsqueeze(1)
            
            # Key: [Batch, Max_Neighbors, 32]
            keys = self.key_proj(nbr_encoded)
            
            # Scores: [Batch, 1, 32] @ [Batch, 32, Max_N] -> [Batch, 1, Max_N]
            scores = torch.bmm(query, keys.transpose(1, 2))
            scores = scores / (32 ** 0.5) # Scale
            weights = F.softmax(scores, dim=-1) # [Batch, 1, Max_N]
            
            # Weighted Sum (Context): [Batch, 1, Max_N] @ [Batch, Max_N, 32] -> [Batch, 1, 32]
            context = torch.bmm(weights, nbr_encoded).squeeze(1)
            
            # Concat
            combined = torch.cat([x_static, context], dim=1)
        else:
            combined = x_static
            
        return self.main_net(combined)

def train_one_model(X, y, seed, epochs=800, lr=0.001):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    
    # Init Model
    model = AttentionWeightedNet(
        static_dim=static_dim, 
        neighbor_dim=dim_vars_all, 
        max_neighbors=MAX_ADJ_NEIGHBORS if USE_ADJ_FEATURES else 0
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for e in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        if torch.isnan(loss): break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    model.eval()
    return model

def predict_ensemble(models, X):
    if len(X) == 0: return np.array([])
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    preds = []
    with torch.no_grad():
        for m in models:
            preds.append(m(X_t).cpu().numpy())
    return np.mean(preds, axis=0)

# ===================== 7) CV 评估 =====================
print("\nRunning CV...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
Y_cv_pred_res = np.zeros_like(Y_train_res_raw)
Base_Y_train_arr = np.array(Base_train_list)
metrics_rows = []

if len(X_train_scaled) >= 5:
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = Y_train_res_raw[train_idx], Y_train_res_raw[val_idx]
        
        models = []
        for i in range(5):
            m = train_one_model(X_tr, y_tr, seed=fold*10+i)
            models.append(m)
        y_pred_val = predict_ensemble(models, X_val)
        Y_cv_pred_res[val_idx] = y_pred_val

    Y_cv_pred_final = np.clip(Base_Y_train_arr + Y_cv_pred_res, 0, 1)
    Y_true_final = Base_Y_train_arr + Y_train_res_raw

    for j, t in enumerate(target_base):
        mae = mean_absolute_error(Y_true_final[:, j], Y_cv_pred_final[:, j])
        rmse = np.sqrt(mean_squared_error(Y_true_final[:, j], Y_cv_pred_final[:, j]))
        r2 = r2_score(Y_true_final[:, j], Y_cv_pred_final[:, j])
        bias = np.mean(Y_cv_pred_final[:, j] - Y_true_final[:, j])
        metrics_rows.append({"Target": t, "MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias})

    print(pd.DataFrame(metrics_rows).round(5))
    try:
        with pd.ExcelWriter(out_cv_xlsx, engine="openpyxl") as writer:
            pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="metrics", index=False)
    except Exception as e: print(f"Error saving CV: {e}")

# ===================== 8) Final Prediction =====================
print("\nFinal Prediction...")
if len(X_pred_scaled) > 0:
    final_models = []
    for i in range(5):
        m = train_one_model(X_train_scaled, Y_train_res_raw, seed=100+i)
        final_models.append(m)

    pred_res = predict_ensemble(final_models, X_pred_scaled)
    Y_pred_final = np.clip(Pred_Base_Y + pred_res, 0, 1)

    completed_df = wide.copy()
    pred_map = {meta["名称"]: Y_pred_final[i] for i, meta in enumerate(Meta_pred_list)}
    
    fill_count = 0
    for idx, row in completed_df.iterrows():
        if row["名称"] in pred_map:
            preds = pred_map[row["名称"]]
            fill_count += 1
            for j, t in enumerate(target_base):
                if pd.isna(completed_df.at[idx, f"{t}_2020"]):
                    completed_df.at[idx, f"{t}_2020"] = preds[j]

    print(f"Filled {fill_count} streets.")
    try:
        with pd.ExcelWriter(out_pred_xlsx, engine="openpyxl") as writer:
            completed_df.to_excel(writer, sheet_name="wide_completed", index=False)
        print(f"Saved to {out_pred_xlsx}")
    except: pass
print("✅ Done. Attention-based Spatial NN.")
