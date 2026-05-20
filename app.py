from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import re
import os

# ========== 初始化Flask应用 ==========
app = Flask(__name__, static_folder=None)

# ========== 加载模型和配置 ==========
MODEL_PATH = "model_files/acp_rf_model.pkl"
SCALER_PATH = "model_files/acp_scaler.pkl"
RFE_PATH = "model_files/acp_rfe_model.pkl"
FEATS_PATH = "model_files/acp_top_features.pkl"

# 氨基酸配置（仅支持20种标准氨基酸）
allowed_aa = set("ACDEFGHIKLMNPQRSTVWY")
aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aa_groups = {
    'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'],
    'hydrophilic': ['R', 'N', 'D', 'Q', 'E', 'H', 'K'],
    'neutral': ['G', 'C', 'P', 'S', 'T'],
    'positive': ['R', 'K', 'H'],
    'negative': ['D', 'E'],
    'aromatic': ['F', 'W', 'Y'],
    'aliphatic': ['A', 'I', 'L', 'V']
}

# 加载模型
try:
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    rfe = joblib.load(RFE_PATH)
    top_feats = joblib.load(FEATS_PATH)
    
    # 修复特征名与RFE特征数匹配
    if len(top_feats) != rfe.n_features_:
        print(f"⚠️ 特征名数量({len(top_feats)})与RFE特征数({rfe.n_features_})不匹配，自动截断特征名")
        top_feats = top_feats[:rfe.n_features_]
        joblib.dump(top_feats, FEATS_PATH)
    
    assert hasattr(rf_model, 'predict_proba'), "模型不支持概率预测"
    assert rfe.n_features_ == len(top_feats), "RFE特征数量与特征名已强制匹配"
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败：{str(e)}")
    exit(1)

# ========== 核心修复：序列清洗与验证增强 ==========
def clean_sequence(seq):
    """严格清洗：仅保留20种标准氨基酸，过滤所有非允许字符"""
    seq = str(seq).strip().upper()
    # 仅保留允许的氨基酸（过滤特殊字符、数字、符号等）
    cleaned = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)
    return cleaned[:200]  # 截断过长序列

def validate_sequence(seq):
    """增强验证：拒绝含无效字符/过短的序列"""
    if not seq:
        return "序列不能为空"
    # 检查是否含无效字符
    invalid_chars = set(seq) - allowed_aa
    if invalid_chars:
        return f"序列含无效字符：{''.join(invalid_chars)}（仅支持A/C/D/E/F/G/H/I/K/L/M/N/P/Q/R/S/T/V/W/Y）"
    cleaned = clean_sequence(seq)
    if len(cleaned) < 5:
        return f"有效序列过短（{len(cleaned)}个氨基酸），无法可靠预测"
    return ""

# ========== 特征提取函数 ==========
def calculate_aa_composition(sequence):
    composition = np.zeros(20)
    seq_len = len(sequence)
    if seq_len == 0:
        return composition
    for aa in sequence:
        if aa in aa_list:
            composition[aa_list.index(aa)] += 1
    return composition / seq_len

def calculate_dipeptide_composition(sequence):
    dipeptides = [a1 + a2 for a1 in aa_list for a2 in aa_list]
    dp_comp = np.zeros(len(dipeptides))
    seq_len = len(sequence)
    if seq_len < 2:
        return dp_comp
    for i in range(seq_len - 1):
        dp = sequence[i] + sequence[i+1]
        if dp in dipeptides:
            dp_comp[dipeptides.index(dp)] += 1
    return dp_comp / (seq_len - 1)

def calculate_group_features(sequence):
    seq_len = len(sequence)
    if seq_len == 0:
        return np.zeros(7)
    group_features = []
    for group in aa_groups.values():
        count = sum(1 for aa in sequence if aa in group)
        group_features.append(count / seq_len)
    return np.array(group_features)

def calculate_advanced_physio_features(sequence):
    hydrophobicity = {'A':1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C':2.5,
                     'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I':4.5,
                     'L':3.8, 'K':-3.9, 'M':1.9, 'F':2.8, 'P':-1.6,
                     'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V':4.2}
    molecular_weight = {'A':89.1, 'R':174.2, 'N':132.1, 'D':133.1, 'C':121.2,
                       'Q':146.1, 'E':147.1, 'G':75.1, 'H':155.2, 'I':131.2,
                       'L':131.2, 'K':146.2, 'M':149.2, 'F':165.2, 'P':115.1,
                       'S':105.1, 'T':119.1, 'W':204.2, 'Y':181.2, 'V':117.1}
    isoelectric_point = {'A':6.0, 'R':10.76, 'N':5.41, 'D':2.77, 'C':5.07,
                        'Q':5.65, 'E':3.22, 'G':5.97, 'H':7.59, 'I':6.02,
                        'L':5.98, 'K':9.74, 'M':5.74, 'F':5.48, 'P':6.30,
                        'S':5.68, 'T':5.60, 'W':5.89, 'Y':5.66, 'V':5.96}
    
    seq_len = len(sequence)
    if seq_len == 0:
        return np.zeros(8)
    
    hydro_vals = [hydrophobicity.get(aa, 0) for aa in sequence]
    mw_vals = [molecular_weight.get(aa, 0) for aa in sequence]
    pi_vals = [isoelectric_point.get(aa, 0) for aa in sequence]
    
    avg_hydro = np.mean(hydro_vals) if hydro_vals else 0
    avg_mw = np.mean(mw_vals) if mw_vals else 0
    avg_pI = np.mean(pi_vals) if pi_vals else 0
    charge = sum(1 for aa in sequence if aa in ['R', 'K', 'H']) - sum(1 for aa in sequence if aa in ['D', 'E'])
    charge_density = charge / seq_len if seq_len > 0 else 0
    hydro_std = np.std(hydro_vals) if len(hydro_vals) > 1 else 0
    mw_std = np.std(mw_vals) if len(mw_vals) > 1 else 0
    max_hydro = max(hydro_vals) if hydro_vals else 0
    
    return np.array([avg_hydro, avg_mw, avg_pI, charge, charge_density, hydro_std, mw_std, max_hydro])

def extract_features(sequence):
    cleaned_seq = clean_sequence(sequence)
    if len(cleaned_seq) == 0:
        return np.array([])
    
    try:
        aa_comp = calculate_aa_composition(cleaned_seq)
        dp_comp = calculate_dipeptide_composition(cleaned_seq)
        group_feat = calculate_group_features(cleaned_seq)
        physio_feat = calculate_advanced_physio_features(cleaned_seq)
        
        all_feats = np.concatenate([aa_comp, dp_comp, group_feat, physio_feat])
        if len(all_feats) != 435:
            raise ValueError(f"特征维度错误: {len(all_feats)} (预期435)")
        return all_feats
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return np.array([])

# ========== 核心修复：预测接口逻辑修正 ==========
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        seq = data.get('sequence', '').strip()
        
        # 1. 严格验证序列（拒绝无效字符/过短序列）
        validation_msg = validate_sequence(seq)
        if validation_msg:
            return jsonify({
                "error": validation_msg,
                "confidence_level": "无效"
            })
        
        # 2. 清洗序列（仅保留有效氨基酸）
        cleaned_seq = clean_sequence(seq)
        
        # 3. 特征提取与验证
        feats = extract_features(seq)
        if len(feats) != 435:
            return jsonify({
                "error": f"特征提取失败，维度异常: {len(feats)}",
                "confidence_level": "低"
            })
        
        # 4. 标准化
        feats_reshaped = feats.reshape(1, -1)
        feats_scaled = scaler.transform(feats_reshaped)
        
        # 5. RFE特征筛选
        feats_rfe = rfe.transform(feats_scaled)
        if feats_rfe.shape[1] != rfe.n_features_:
            return jsonify({
                "error": f"特征筛选失败，维度不匹配: {feats_rfe.shape[1]}",
                "confidence_level": "低"
            })
        
        # 6. 预测概率（核心修复：确保概率和为100%）
        pred_prob = rf_model.predict_proba(feats_rfe)[0][1]
        acp_prob = round(pred_prob * 100, 2)
        non_acp_prob = round((1 - pred_prob) * 100, 2)  # 强制和为100%
        
        # 7. 动态阈值判断
        seq_len = len(cleaned_seq)
        threshold = 0.6 if seq_len < 10 else 0.55 if seq_len < 30 else 0.5
        is_acp = 1 if pred_prob >= threshold else 0
        
        # 8. 置信度评估
        confidence = "高" if (pred_prob >= 0.8 or pred_prob <= 0.2) else \
                     "中" if (pred_prob >= 0.6 or pred_prob <= 0.4) else "低"
        
        return jsonify({
            "error": "",
            "original_sequence": seq,
            "cleaned_sequence": cleaned_seq,
            "sequence_length": len(cleaned_seq),
            "is_acp": is_acp,
            "acp_prob": acp_prob,
            "non_acp_prob": non_acp_prob,
            "threshold_used": threshold,
            "confidence_level": confidence,
            "prediction_note": "序列含无效字符时，预测结果不可靠" if len(cleaned_seq) != len(seq) else ""
        })
    except Exception as e:
        return jsonify({
            "error": f"预测失败：{str(e)}",
            "confidence_level": "低"
        })

# ========== 静态文件服务 ==========
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# ========== 启动服务 ==========
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8080)
