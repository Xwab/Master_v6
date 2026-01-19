#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================== 0) 配置 =====================
# 请根据实际情况修改路径
excel_path = "/home/users/yyb/last/567append_all.xlsx"
sheet_name = "Sheet1"

# === 开关: 是否使用 KNN 邻居特征作为输入 ===
USE_KNN_FEATURES = True

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
    # 生成假数据用于演示 (如果文件不存在)
    print("Generating dummy data for testing logic...")
    data = []
    names = [f"Street_{i}" for i in range(100)]
    for y in [2000, 2010, 2020]:
        for n in names:
            # 模拟部分缺失: 2020年部分target缺失
            is_target_missing = (y == 2020) and (int(n.split('_')[1]) > 70)
            row = {"名称": n, "year": y, "area": 100}
            for v in ["age65sacle", "malesacle", "nong_incomesacle", "shengchan_incomesacle", 
                      "qita_incomesacle", "unemploysacle", "tizhisacle", "outsidesacle"]:
                row[v] = np.random.rand()
            for t in ["bungalowsacle", "nowcsacle", "nobathroomsacle", "nokitchensacle"]:
                row[t] = np.nan if is_target_missing else np.random.rand()
            # Non-sacle columns (not used in main logic but present in load)
            for t in ["bungalow", "nowc", "nobathroom", "nokitchen"]:
                row[t] = np.nan
            data.append(row)
    df = pd.DataFrame(data)

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["名称"] = df["名称"].astype(str).str.strip()

# 定义变量列表
# 目标变量 (Percentages)
target_base = ["bungalowsacle", "nowcsacle", "nobathroomsacle", "nokitchensacle"]
# 原始计数变量 (用于清洗，不直接入模)
target_base_no_sacle = ["bungalow", "nowc", "nobathroom", "nokitchen"]

# 所有入模变量 (包含 Feature Base + Target Base)
# 注意: feature_base 是指除目标外的特征 (人口、经济等)
vars_all = [
    "area", "age65sacle", "malesacle",
    "nong_incomesacle", "shengchan_incomesacle", "qita_incomesacle",
    "unemploysacle", "tizhisacle", "outsidesacle",
    "bungalowsacle", "nowcsacle", "nobathroomsacle", "nokitchensacle"
    # 注意: "bungalow" 等原始计数列不包含在这里，避免混淆
]

# feature_base: 用作 Input 的特征 (不含目标变量)
feature_base = [v for v in vars_all if v not in target_base]

# 类型转换
cols_to_numeric = vars_all + target_base_no_sacle
for c in cols_to_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Wide转换: index=名称, columns=year
# 为了方便处理，我们先 pivot
wide = df.pivot_table(index="名称", columns="year", values=cols_to_numeric, aggfunc="first")
# Flatten columns
wide.columns = [f"{v}_{int(y)}" for (v, y) in wide.columns]
wide = wide.reset_index()

print("Data Loaded & Pivoted. Wide shape:", wide.shape)

# ===================== 2) 筛选连续存在的街道/乡镇 =====================
# 逻辑：
# set1: 在 2000 和 2010 都存在的
# set2: 在 2010 和 2020 都存在的
# 注意：一个街道可能同时属于 set1 和 set2，这没问题，它会贡献两个样本（分别用于两个时段）

def get_valid_names(df_wide, y1, y2, check_cols_prefix):
    """
    检查指定年份是否有数据。
    这里只要 check_cols_prefix 在对应年份列非全空即可判定为"存在"。
    或者简单点，只要 pivot 后有对应的列且该行不全为 NaN。
    由于 pivot fill_value=NaN，我们检查关键列是否存在。
    """
    # 构建该年份必须存在的列名列表 (取 feature_base 中的一个代表即可，或者检查所有)
    # 只要 feature_base 中的数据大部分存在即可。为了严格起见，检查所有 feature_base
    
    # 简单的存在性检查：检查 feature_base 在 y1 和 y2 是否大体完整
    # 这里放宽一点：只要 feature_base 的列在 dataframe 里有值 (not na)
    
    valid_indices = []
    for idx, row in df_wide.iterrows():
        # 检查 y1
        has_y1 = all(pd.notna(row.get(f"{v}_{y1}", np.nan)) for v in feature_base)
        # 检查 y2
        has_y2 = all(pd.notna(row.get(f"{v}_{y2}", np.nan)) for v in feature_base)
        
        if has_y1 and has_y2:
            valid_indices.append(row["名称"])
            
    return set(valid_indices)

names_00_10 = get_valid_names(wide, 2000, 2010, feature_base)
names_10_20 = get_valid_names(wide, 2010, 2020, feature_base)

print(f"Valid pairs 2000-2010: {len(names_00_10)}")
print(f"Valid pairs 2010-2020: {len(names_10_20)}")

# ===================== 3) 定义处理函数：处理一个时间段 (T_prev -> T_curr) =====================
def process_period(wide_df, names_subset, year_prev, year_curr, is_training_data=True):
    """
    对指定名单 names_subset 和年份对 (year_prev, year_curr) 进行处理：
    1. 构建 KNN 搜索空间 (Search Vector: All_Vars_Prev + Feature_Base_Curr)
    2. 计算 KNN 邻居均值
    3. 构建样本特征 (Vars_Prev, Feature_Base_Curr, Delta, Neighbor_Feats)
    4. 构建目标 (Residuals)
    """
    
    # 1. 筛选数据
    # 只取在该时间段存在的行
    sub_df = wide_df[wide_df["名称"].isin(names_subset)].copy()
    if sub_df.empty:
        return [], [], [], []
    
    sub_df = sub_df.reset_index(drop=True)
    
    # 2. 准备 KNN 搜索特征
    # 搜索向量 = [所有特征_{Prev}, 非住房特征_{Curr}]
    # 理由：用"之前的全貌" + "现在的非目标特征" 来寻找相似街道
    search_cols = [f"{v}_{year_prev}" for v in vars_all] + \
                  [f"{v}_{year_curr}" for v in feature_base]
    
    # 检查列是否存在 (防止某些列在 pivot 时丢失)
    missing_cols = [c for c in search_cols if c not in sub_df.columns]
    if missing_cols:
        # 如果列不存在，填充NaN
        for c in missing_cols:
            sub_df[c] = np.nan
            
    X_search = sub_df[search_cols].values
    
    # 填充 + 标准化 (用于 KNN)
    imputer_knn = SimpleImputer(strategy="median")
    X_search_imputed = imputer_knn.fit_transform(X_search)
    
    scaler_knn = StandardScaler()
    X_search_scaled = scaler_knn.fit_transform(X_search_imputed)
    X_search_scaled = np.nan_to_num(X_search_scaled, nan=0.0)
    
    # Fit KNN
    # n_neighbors 取 11 (包含自己)
    k = min(11, len(sub_df))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_search_scaled)
    _, indices = nbrs.kneighbors(X_search_scaled)
    
    # 3. 构建样本
    X_samples = []
    Y_residuals = []
    Base_values = []
    Meta_info = [] # 存储名称等
    
    # 预先获取列索引以加速
    col_idx_map = {c: i for i, c in enumerate(sub_df.columns)}
    
    # 准备数据矩阵
    data_matrix = sub_df.values
    
    for i in range(len(sub_df)):
        name = sub_df.iloc[i]["名称"]
        
        # --- KNN 特征计算 ---
        neighbor_idxs = indices[i, 1:] # 排除自己
        if len(neighbor_idxs) == 0:
             # Fallback if k is small (e.g. only 1 sample)
            neighbor_idxs = [i]

        nbr_feats = []
        
        # 邻居特征 1: Prev Year Mean (All Vars)
        # 计算邻居在 year_prev 的 vars_all 均值
        for v in vars_all:
            col_name = f"{v}_{year_prev}"
            if col_name in col_idx_map:
                vals = data_matrix[neighbor_idxs, col_idx_map[col_name]]
                nbr_feats.append(np.nanmean(vals.astype(float)))
            else:
                nbr_feats.append(0.0)
                
        # 邻居特征 2: Curr Year Mean (Feature Base Only)
        # 计算邻居在 year_curr 的 feature_base 均值
        # 注意：不使用邻居的 target (housing) _curr，以保持一致性(预测时也没有)
        # 除非确定作为 spatial lag。这里根据"用10年前...计算KNN...求KNN邻居均值"，
        # 保守起见使用 input features 的均值。
        for v in feature_base:
            col_name = f"{v}_{year_curr}"
            if col_name in col_idx_map:
                vals = data_matrix[neighbor_idxs, col_idx_map[col_name]]
                nbr_feats.append(np.nanmean(vals.astype(float)))
            else:
                nbr_feats.append(0.0)
        
        # --- 样本自身特征 ---
        sample_feats = []
        
        # A. 历史全量 (Prev)
        prev_vals_dict = {}
        for v in vars_all:
            col_name = f"{v}_{year_prev}"
            val = sub_df.at[i, col_name] if col_name in sub_df.columns else np.nan
            sample_feats.append(val)
            prev_vals_dict[v] = val
            
        # B. 当前非目标 (Curr)
        curr_vals_dict = {}
        for v in feature_base:
            col_name = f"{v}_{year_curr}"
            val = sub_df.at[i, col_name] if col_name in sub_df.columns else np.nan
            sample_feats.append(val)
            curr_vals_dict[v] = val
            
        # C. 变化量 (Delta)
        for v in feature_base:
            p = prev_vals_dict.get(v, np.nan)
            c = curr_vals_dict.get(v, np.nan)
            if pd.notna(p) and pd.notna(c):
                sample_feats.append(c - p)
            else:
                sample_feats.append(np.nan)
                
        # D. KNN 邻居特征
        if USE_KNN_FEATURES:
            sample_feats.extend(nbr_feats)
        
        # --- Target (Residuals) ---
        # Base value = Prev year target
        base_vals = []
        for t in target_base:
            base_vals.append(prev_vals_dict.get(t, np.nan))
            
        # Residual = Curr - Prev
        # 仅当我们需要 label 时计算 (training data)
        # 且必须当前也有值
        residuals = []
        has_labels = True
        
        if is_training_data: # 或者是构建 potential training sample
            # 检查是否有 Target Label
            for idx_t, t in enumerate(target_base):
                # 修改：根据 raw count (target_base_no_sacle) 是否为 NaN 来判断是否缺失
                # 注意：target_base_no_sacle 必须与 target_base 顺序对应
                t_raw = target_base_no_sacle[idx_t]
                col_name_raw = f"{t_raw}_{year_curr}"
                val_raw = sub_df.at[i, col_name_raw] if col_name_raw in sub_df.columns else np.nan

                col_name = f"{t}_{year_curr}"
                val_curr = sub_df.at[i, col_name] if col_name in sub_df.columns else np.nan
                val_prev = base_vals[idx_t]
                
                # 判定条件：Raw Count 非空 (表示当前年份数据有效，哪怕 val_curr 是 0) 且 Prev 非空
                if pd.notna(val_raw) and pd.notna(val_prev):
                    residuals.append(val_curr - val_prev)
                else:
                    residuals.append(np.nan)
                    has_labels = False # 只要有一个target缺失，就视为无完整label
        
        # 收集结果
        # 注意：预测集不需要 has_labels 为真
        X_samples.append(sample_feats)
        Y_residuals.append(residuals if is_training_data else [np.nan]*4)
        Base_values.append(base_vals)
        Meta_info.append({"名称": name, "Period": f"{year_prev}->{year_curr}", "HasLabel": has_labels})

    return X_samples, Y_residuals, Base_values, Meta_info


# ===================== 4) 构建数据集 =====================
print("Processing Period 1: 2000 -> 2010...")
X1, Y1, Base1, Meta1 = process_period(wide, names_00_10, 2000, 2010, is_training_data=True)

print("Processing Period 2: 2010 -> 2020...")
# 这里我们对所有 2010-2020 存在的街道都进行处理，之后再根据是否有 Label 分为 Train/Pred
X2, Y2, Base2, Meta2 = process_period(wide, names_10_20, 2010, 2020, is_training_data=True)

# 合并并拆分 Train / Pred
X_train_list = []
Y_train_list = []
Base_train_list = []
Meta_train_list = []

X_pred_list = []
Base_pred_list = []
Meta_pred_list = []

# 处理 Period 1 (2000->2010): 只要有 Label 就进训练集
for i, meta in enumerate(Meta1):
    if meta["HasLabel"] and not np.isnan(Y1[i]).any():
        X_train_list.append(X1[i])
        Y_train_list.append(Y1[i])
        Base_train_list.append(Base1[i])
        Meta_train_list.append(meta)

# 处理 Period 2 (2010->2020): 有 Label -> 训练集, 无 Label -> 预测集
for i, meta in enumerate(Meta2):
    if meta["HasLabel"] and not np.isnan(Y2[i]).any():
        X_train_list.append(X2[i])
        Y_train_list.append(Y2[i])
        Base_train_list.append(Base2[i])
        Meta_train_list.append(meta)
    else:
        # 如果是 2010->2020 且缺失 Label，则是我们需要预测的目标
        # 注意：这里我们假设这就是目标任务 (预测 2020 缺失值)
        X_pred_list.append(X2[i])
        Base_pred_list.append(Base2[i])
        Meta_pred_list.append(meta)

X_train_raw = np.array(X_train_list, dtype=np.float32)
Y_train_res_raw = np.array(Y_train_list, dtype=np.float32)

X_pred_raw = np.array(X_pred_list, dtype=np.float32)
Pred_Base_Y = np.array(Base_pred_list, dtype=np.float32)

print(f"Dataset Constructed.")
print(f"Using KNN Features: {USE_KNN_FEATURES}")
print(f"Train samples: {len(X_train_raw)} (From 2000->2010 and 2010->2020)")
print(f"Pred samples:  {len(X_pred_raw)} (From 2010->2020 Missing Labels)")

# ===================== 5) 预处理 (Impute + Scale) =====================
imputer = SimpleImputer(strategy="median")
# 训练集 fit
X_train_imputed = imputer.fit_transform(X_train_raw)
# 预测集 transform
if len(X_pred_raw) > 0:
    X_pred_imputed = imputer.transform(X_pred_raw)
else:
    X_pred_imputed = X_pred_raw

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
if len(X_pred_raw) > 0:
    X_pred_scaled = scaler.transform(X_pred_imputed)
else:
    X_pred_scaled = X_pred_raw

# ===================== 6) 模型定义 =====================
class RobustNet(nn.Module):
    def __init__(self, input_dim):
        super(RobustNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        return self.net(x)

def train_one_model(X, y, seed, epochs=800, lr=0.001):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    
    model = RobustNet(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for e in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        
        if torch.isnan(loss):
            print(f"⚠️ NaN Loss at epoch {e}. Break.")
            break
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    model.eval()
    return model

def predict_ensemble(models, X):
    if len(X) == 0:
        return np.array([])
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

# 记录 CV 结果
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

    # 还原
    Y_cv_pred_final = Base_Y_train_arr + Y_cv_pred_res
    Y_cv_pred_final = np.clip(Y_cv_pred_final, 0, 1)
    Y_true_final = Base_Y_train_arr + Y_train_res_raw

    for j, t in enumerate(target_base):
        mae = mean_absolute_error(Y_true_final[:, j], Y_cv_pred_final[:, j])
        rmse = np.sqrt(mean_squared_error(Y_true_final[:, j], Y_cv_pred_final[:, j]))
        r2 = r2_score(Y_true_final[:, j], Y_cv_pred_final[:, j])
        bias = np.mean(Y_cv_pred_final[:, j] - Y_true_final[:, j])
        
        metrics_rows.append({"Target": t, "MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias})

    print(pd.DataFrame(metrics_rows).round(5))
    
    # 保存 CV 结果
    try:
        with pd.ExcelWriter(out_cv_xlsx, engine="openpyxl") as writer:
            pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="metrics", index=False)
            # 保存详细预测对比
            cv_detail = pd.DataFrame(Y_cv_pred_final, columns=[f"Pred_{t}" for t in target_base])
            for j, t in enumerate(target_base):
                cv_detail[f"True_{t}"] = Y_true_final[:, j]
            # 添加 Meta 信息
            meta_df = pd.DataFrame(Meta_train_list)
            cv_detail = pd.concat([meta_df.reset_index(drop=True), cv_detail], axis=1)
            cv_detail.to_excel(writer, sheet_name="cv_details", index=False)
    except Exception as e:
        print(f"Error saving CV results: {e}")
else:
    print("Not enough samples for CV.")

# ===================== 8) 最终补全 (Prediction) =====================
print("\nFinal Prediction on missing 2020 data...")

if len(X_pred_scaled) > 0:
    final_models = []
    for i in range(5):
        m = train_one_model(X_train_scaled, Y_train_res_raw, seed=100+i)
        final_models.append(m)

    pred_res = predict_ensemble(final_models, X_pred_scaled)

    # 还原: Y_2020 = Y_2010 (Base) + Delta
    Y_pred_final = Pred_Base_Y + pred_res
    Y_pred_final = np.clip(Y_pred_final, 0, 1)

    # 填回原始 Wide 表 (或者创建一个新的结果表)
    # 我们基于 wide 表做一份拷贝
    completed_df = wide.copy()
    
    # 构建预测结果字典: {名称: [pred_val1, pred_val2, ...]}
    pred_map = {}
    for i, meta in enumerate(Meta_pred_list):
        name = meta["名称"]
        pred_map[name] = Y_pred_final[i]
        
    # 填充
    fill_count = 0
    for idx, row in completed_df.iterrows():
        name = row["名称"]
        if name in pred_map:
            preds = pred_map[name]
            fill_count += 1
            for j, t in enumerate(target_base):
                col_2020 = f"{t}_2020"
                # 即使原位置有值(理论上不应该，因为我们在Pred list里)，也覆盖或填充
                # 根据题意是填充缺失的
                if pd.isna(completed_df.at[idx, col_2020]):
                    completed_df.at[idx, col_2020] = preds[j]

    print(f"Filled {fill_count} streets with predictions.")

    try:
        with pd.ExcelWriter(out_pred_xlsx, engine="openpyxl") as writer:
            completed_df.to_excel(writer, sheet_name="wide_completed", index=False)
        print(f"Saved completed file to {out_pred_xlsx}")
    except Exception as e:
        print(f"Error saving prediction results: {e}")

else:
    print("No prediction samples found.")

print("✅ Done. Continuity-Filtered KNN + Residual NN Ensemble.")
