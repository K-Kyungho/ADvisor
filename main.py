# =======================
# Imports & Settings
# =======================
import os, re, json, base64, random, requests, gc, argparse, pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- OpenAI API (Chat Completions via requests, with vision) ---
API_URL_OPENAI = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = "YOUR API KEY HERE"
HEADERS_OPENAI = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

SPEND_COL = "spend"
CLICKS_COL = "clicks"
IMP_COL = "impressions"

CSV_DIR = "data/"

CAPTION_MAP: Dict[str, str] = {}

METRIC_MODES: Dict[str, Dict[str, str]] = {
    "spc": {"col": "SpendPerClick", "direction": "lower"},
    "ctr": {"col": "CTR", "direction": "higher"},
    "spi": {"col": "SpendPerImpression", "direction": "lower"},
}

TABULAR_COLS_NUM = [] 
TABULAR_COLS_CAT = []

BRAND_SELECTED_FEATURES: Dict[str, Dict[str, str]] = {}  # {brand_id: {feature_key: description}}
BRAND_DESCRIPTIONS: Dict[str, str] = {}  # {brand_id: description}
IMG_ROOT = "path to image data"
IMG_TEMPLATE = "{ad_id}.jpg"


OPENAI_MODEL_NAME = "gpt-4.1-mini"  # You can change the model.

FEWSHOT_MAX_EXAMPLES = 4      
FEWSHOT_EXAMPLE_SIZE = 4    

LLM_BATCH_SIZE = 16            
  
### utils functions
def load_caption_map(path: str) -> Dict[str, str]:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        out[str(k)] = (v or "")
    print(f"[Caption] Loaded {len(out)} captions from {path}")
    return out

def encode_image_to_dataurl(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def row_to_mm_dict(row: pd.Series) -> Dict[str, Any]:
    global CAPTION_MAP

    ad_id = str(row["ad_id"])
    caption = None

    if "caption" in row and isinstance(row["caption"], str) and row["caption"].strip():
        caption = row["caption"]
    else:
        if CAPTION_MAP:
            caption = CAPTION_MAP.get(ad_id, "")

    if caption is None:
        caption = ""

    d = {"ad_id": ad_id, "caption": caption[:800]}
    img_path = os.path.join(IMG_ROOT, IMG_TEMPLATE.format(ad_id=ad_id))
    d["image_data_url"] = encode_image_to_dataurl(img_path) if os.path.exists(img_path) else None
    return d

def make_relevance_grades(y: pd.Series, n_bins: int = 5, direction: str = "higher") -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    if y.nunique(dropna=True) <= 1:
        return pd.Series([0] * len(y), index=y.index, dtype=int)
    
    ranked = y.rank(method="average")
    if direction == "lower":
        ranked = len(ranked) + 1 - ranked
    
    q = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
    q = q.fillna(0).astype(int)
    return q

def eval_ndcg_kendall(df_pred: pd.DataFrame,  df_truth: pd.DataFrame,  target_col: str,  direction: str = "higher") -> Dict[str, float]:
    if df_pred is None or df_pred.empty:
        return {
            "ndcg@1": np.nan,
            "ndcg@3": np.nan,
            "ndcg@5": np.nan
        }

    df_p = df_pred.dropna(subset=["ad_id", "score"]).copy()
    df_p["ad_id"] = df_p["ad_id"].astype(str)
    df_p = df_p.groupby("ad_id", as_index=False)["score"].mean()

    df_t = df_truth.dropna(subset=["ad_id", target_col]).copy()
    df_t["ad_id"] = df_t["ad_id"].astype(str)
    
    m = pd.merge(df_p, df_t[["ad_id", target_col]], on="ad_id", how="inner")
    if m.empty:
        return {
            "ndcg@1": np.nan,
            "ndcg@3": np.nan,
            "ndcg@5": np.nan
        }

    if direction == "lower":
        y_true_vals = np.max(m[target_col].values) + 1 - m[target_col].values
    else:
        y_true_vals = m[target_col].values

    y_true = np.array([y_true_vals], dtype=float)
    y_score = np.array([m["score"].values], dtype=float)

    ks = [1, 3, 5]
    ndcgs = {}
    for k in ks:
        try:
            score = float(ndcg_score(y_true, y_score, k=k))
            ndcgs[f"ndcg@{k}"] = score if not np.isnan(score) else 0.0
        except Exception as e:
            print(f"[NDCG ERROR] k={k}, exception: {e}")
            ndcgs[f"ndcg@{k}"] = 0.0

    return {
        **ndcgs,
    }

def add_target_metric(df: pd.DataFrame, clicks_col: str, spend_col: str, imp_col: str, mode: str, out_col: str) -> pd.DataFrame:
    df2 = df.copy()
    df2[clicks_col] = pd.to_numeric(df2.get(clicks_col), errors="coerce")
    df2[spend_col] = pd.to_numeric(df2.get(spend_col), errors="coerce")
    if imp_col in df2.columns:
        df2[imp_col] = pd.to_numeric(df2.get(imp_col), errors="coerce")

    if mode == "spc":
        denom = df2[clicks_col].replace(0, np.nan)
        ratio = df2[spend_col] / denom
    elif mode == "ctr":
        denom = df2[imp_col].replace(0, np.nan)
        ratio = df2[clicks_col] / denom
    elif mode == "spi":
        denom = df2[imp_col].replace(0, np.nan)
        ratio = df2[spend_col] / denom
    else:
        raise ValueError(f"Unsupported TARGET_MODE: {mode}")

    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df2[out_col] = ratio.astype(float)
    return df2

# =======================
# For Step 1: Similar Brand Selection
# =======================
def load_embeddings(embedding_path: str) -> Dict[str, np.ndarray]:
    with open(embedding_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def find_most_similar_brand(target_brand_id: str, all_metadata: pd.DataFrame, metric_modes: List[str], all_embeddings: Dict[str, np.ndarray] = None,    brand_embeddings: Dict[str, np.ndarray] = None) -> Tuple[str, List[Dict[str, Any]]]:

    if all_metadata.empty:
        return None, []
    
    use_brand_emb = brand_embeddings is not None and len(brand_embeddings) > 0
    
    target_brand_id_str = str(target_brand_id)
    
    if use_brand_emb:
        target_emb = brand_embeddings[target_brand_id_str]
        print(f"[Cross-Brand] Using pre-computed brand embedding for {target_brand_id}")
        
    other_brands = all_metadata[all_metadata["brand_id"].astype(str) != target_brand_id_str]["brand_id"].unique()
    
    if len(other_brands) == 0:
        return None, []
    
    brand_similarities = {}
    for bid in other_brands:
        bid_str = str(bid)
        
        if use_brand_emb:
            if bid_str not in brand_embeddings:
                continue
            
            brand_emb = brand_embeddings[bid_str]
        
        similarity = np.dot(target_emb, brand_emb) / (
            np.linalg.norm(target_emb) * np.linalg.norm(brand_emb) + 1e-8
        )
        brand_similarities[bid_str] = similarity
    
    if not brand_similarities:
        print(f"[Cross-Brand] No similar brands found")
        return None, []
    
    similar_brand_id = max(brand_similarities, key=brand_similarities.get)
    sim_score = brand_similarities[similar_brand_id]
    print(f"[Cross-Brand] Most similar brand: {similar_brand_id} (cosine sim: {sim_score:.4f})")
    
    # Threshold check for brand selection
    if sim_score < 0.6:
        print(f"[Cross-Brand] Similarity {sim_score:.4f} is below threshold 0.6, not using cross-brand data")
        return None, []
    
    similar_brand_data = all_metadata[all_metadata["brand_id"].astype(str) == str(similar_brand_id)].copy()
    
    if similar_brand_data.empty:
        return None, []
    
    primary_metric = metric_modes[0] if metric_modes else "ctr"
    metric_col = METRIC_MODES.get(primary_metric, {}).get("col", "CTR")
    
    if metric_col not in similar_brand_data.columns:
        similar_brand_data = add_target_metric(similar_brand_data, clicks_col=CLICKS_COL, spend_col=SPEND_COL, imp_col=IMP_COL, mode=primary_metric,out_col=metric_col)
    
    similar_brand_data[metric_col] = pd.to_numeric(similar_brand_data[metric_col], errors="coerce")
    similar_brand_data = similar_brand_data.dropna(subset=[metric_col])
    
    if similar_brand_data.empty:
        return None, []
    
    # Few-shot Construction: Top 2, Mid 2, Bottom 2
    direction = METRIC_MODES[primary_metric].get("direction", "higher")
    if direction == "higher":
        sorted_data = similar_brand_data.sort_values(metric_col, ascending=False)
    else:
        sorted_data = similar_brand_data.sort_values(metric_col, ascending=True)
    
    reasoning_samples = []
    n = len(sorted_data)
    
    # Top 2
    for idx in sorted_data.iloc[:2].index:
        reasoning_samples.append({
            "row": sorted_data.loc[idx],
            "performance_tier": "high",
        })
    
    # Middle 2
    mid_start = max(0, n // 3)
    mid_end = min(n, n * 2 // 3)
    if mid_start < mid_end:
        for idx in sorted_data.iloc[mid_start:mid_end][:2].index:  
            reasoning_samples.append({
                "row": sorted_data.loc[idx],
                "performance_tier": "medium",
            })
    
    # Bottom 2
    for idx in sorted_data.iloc[-2:].index:
        reasoning_samples.append({
            "row": sorted_data.loc[idx],
            "performance_tier": "low",
        })
    
    return str(similar_brand_id), reasoning_samples[:6]  

def extract_brand_performance_samples(brand_df: pd.DataFrame,  metric_modes: List[str],  num_per_tier: int = 2) -> List[Dict[str, Any]]:
    if brand_df.empty or not metric_modes:
        return []
    
    primary_metric = metric_modes[0]
    metric_col = METRIC_MODES.get(primary_metric, {}).get("col", "CTR")
    
    if metric_col not in brand_df.columns:
        brand_df = add_target_metric(brand_df, clicks_col=CLICKS_COL, spend_col=SPEND_COL, imp_col=IMP_COL, mode=primary_metric, out_col=metric_col)        
    
    brand_df[metric_col] = pd.to_numeric(brand_df[metric_col], errors="coerce")
    brand_df = brand_df.dropna(subset=[metric_col])
    
    if brand_df.empty:
        return []
    
    direction = METRIC_MODES[primary_metric].get("direction", "higher")
    if direction == "higher":
        sorted_data = brand_df.sort_values(metric_col, ascending=False)
    else:
        sorted_data = brand_df.sort_values(metric_col, ascending=True)
    
    performance_samples = []
    n = len(sorted_data)
    
    # Top samples (high performance)
    for idx in sorted_data.iloc[:num_per_tier].index:
        performance_samples.append({
            "row": sorted_data.loc[idx],
            "performance_tier": "high",
        })
    
    # Middle samples (medium performance)
    mid_start = max(0, n // 3)
    mid_end = min(n, n * 2 // 3)
    if mid_start < mid_end:
        for idx in sorted_data.iloc[mid_start:mid_end][:num_per_tier].index:
            performance_samples.append({
                "row": sorted_data.loc[idx],
                "performance_tier": "medium",
            })
    
    # Bottom samples (low performance)
    for idx in sorted_data.iloc[-num_per_tier:].index:
        performance_samples.append({
            "row": sorted_data.loc[idx],
            "performance_tier": "low",
        })
    
    return performance_samples[:num_per_tier * 3]

def build_cross_brand_reasoning(
    similar_brand_id: str,
    reasoning_samples: List[Dict[str, Any]],
    metric_modes: List[str],
) -> str:
    """
    Similar brand의 샘플들을 바탕으로 LLM에게 scoring criteria를 reasoning하도록 요청.
    Image + Caption을 함께 제시.
    
    Returns: LLM이 생성한 reasoning text
    """
    if not reasoning_samples or not metric_modes:
        return ""
    
    primary_metric = metric_modes[0]
    metric_col = METRIC_MODES[primary_metric]["col"]
    direction = METRIC_MODES[primary_metric]["direction"]
    
    system_prompt = (
        "You are an expert in Instagram ad performance analysis. "
        "You will analyze real ad performance data (caption + image) and infer scoring criteria."
    )
    
    # Multimodal content 구성 (text + images)
    content: List[Dict[str, Any]] = []
    content.append({
        "type": "text",
        "text": f"I'm analyzing ads from Brand {similar_brand_id}. Below are real examples with actual performance metrics:\n\n"
    })
    
    # 각 샘플을 caption + image로 제시
    for i, sample in enumerate(reasoning_samples, 1):
        row = sample["row"]
        tier = sample["performance_tier"]
        ad_id = row.get("ad_id", "N/A")
        metric_val = row.get(metric_col, "N/A")
        caption = row.get("caption", "No caption")
        
        # 텍스트 설명
        content.append({
            "type": "text",
            "text": f"\n{i}. [{tier.upper()}] Ad {ad_id}: {metric_col}={metric_val:.4f}\nCaption: {caption}\n"
        })
        
        # Image 추가
        mm_dict = row_to_mm_dict(row)
        if mm_dict.get("image_data_url"):
            content.append({
                "type": "image_url",
                "image_url": {"url": mm_dict["image_data_url"]}
            })
    
    # 질문 추가
    content.append({
        "type": "text",
        "text": (
            f"\n\nBased on these examples (both captions AND images), please reason about:\n"
            f"1. What visual/textual characteristics correlate with HIGH vs LOW {primary_metric.upper()}?\n"
            f"2. What specific patterns do you see in the high-performing ads?\n\n"
            f"Please provide 3-4 sentences of reasoning that could help score similar ads."
        )
    })
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0
    }
    
    try:
        resp = requests.post(API_URL_OPENAI, headers=HEADERS_OPENAI, json=payload, timeout=180)
        if resp.status_code != 200:
            return ""
        
        reasoning_text = resp.json()["choices"][0]["message"]["content"].strip()
        return reasoning_text
    except Exception as e:
        print(f"[Cross-Brand] Failed to get reasoning: {e}")
        return ""

def select_features_for_brand(brand_id: str,
    num_features: int = 4,
    brand_description: str = None,
    cross_brand_reasoning: str = None,
    brand_performance_samples: List[Dict[str, Any]] = None,
    metric_modes: List[str] = None,
    fewshot_examples: List[Dict[str, Any]] = None,
) -> Dict[str, str]:

    # Few-shot examples context 구성
    primary_metric = metric_modes[0]
    metric_col = METRIC_MODES.get(primary_metric, {}).get("col", "CTR")
    
    fewshot_text = [
        "PERFORMANCE RANKING EXAMPLES (for feature selection):",
        "="*60,
        f"Metric Focus: {primary_metric.upper()} ({metric_col})",
        f"Direction: {METRIC_MODES[primary_metric].get('direction', 'higher')} is better",
        "\nBelow are groups of ranked ads. Analyze patterns to identify what makes ads perform well.",
        "Each group shows ADS RANKED FROM HIGH TO LOW PERFORMERS in this metric.",
        ""
    ]
    
    for ex_idx, example in enumerate(fewshot_examples[:3], 1):
        ads = example.get("ads", [])
        if not ads:
            continue
            
        fewshot_text.append(f"\nRanking Group {ex_idx}:")
        for rank_idx, ad in enumerate(ads[:5], 1):
            ad_id = ad.get("ad_id", "N/A")
            caption = ad.get("caption", "No caption")[:100]
            metric_val = ad.get("metric_val", "N/A")
            fewshot_text.append(f"  {rank_idx}. Ad {ad_id}: {caption}... ({metric_col}={metric_val})")
    
    fewshot_context = "\n".join(fewshot_text)
    
    brand_context = ""
    if brand_description:
        brand_context = f"\n\nBrand Context (ID: {brand_id}):\n{brand_description}\n"
    else:
        brand_context = f"\n\nBrand ID: {brand_id}\n"
    
    reasoning_context = ""
    if cross_brand_reasoning:
        reasoning_context = ("\n\nSCORING INSIGHTS FROM SIMILAR BRANDS:\n"f"{cross_brand_reasoning}\n")
    
    brand_samples_content: List[Dict[str, Any]] = []
    if brand_performance_samples and metric_modes:
        brand_samples_text = [
            "\n\nADDITIONAL BRAND PERFORMANCE SAMPLES:",
            f"Real ads from THIS brand ({brand_id}) showing high vs low performers.\n"
        ]
        
        for i, sample in enumerate(brand_performance_samples, 1):
            row = sample["row"]
            tier = sample["performance_tier"]
            ad_id = row.get("ad_id", "N/A")
            metric_val = row.get(metric_col, "N/A")
            caption = row.get("caption", "No caption")[:80]
            
            brand_samples_text.append(
                f"{i}. [{tier.upper()}] Ad {ad_id}: {metric_col}={metric_val:.4f} | {caption}"
            )
        
        brand_samples_content.append({
            "type": "text",
            "text": "\n".join(brand_samples_text)
        })
        
        img_count = 0
        for sample in brand_performance_samples:
            if img_count >= 3:
                break
            mm_dict = row_to_mm_dict(sample["row"])
            if mm_dict.get("image_data_url") and mm_dict["image_data_url"] is not None:
                brand_samples_content.append({
                    "type": "image_url",
                    "image_url": {"url": mm_dict["image_data_url"]}
                })
                img_count += 1
    
    system_prompt = (
        "You are an expert in brand marketing and ad performance optimization.\n"
        "Your task is to generate and prioritize the most important features for evaluating ads.\n\n"
        "CRITICAL: All generated features MUST be:\n"
        "1. Numeric, scoreable on a 1-5 integer scale\n"
        "2. Distinct and meaningful for ad evaluation\n"
        "3. Based on patterns observed in the ranking examples provided"
    )
    
    user_prompt_text = (
        f"{fewshot_context}"
        f"{brand_context}"
        f"{reasoning_context}"
        f"\nBased on the ranking examples above, identify and prioritize the {num_features} MOST IMPORTANT features for evaluating ads for this brand.\n\n"
        f"For each feature, provide:\n"
        f"1. Feature name/key (concise, lowercase with underscores, e.g., 'visual_quality', 'brand_resonance')\n"
        f"2. Why this feature is critical based on the patterns you observe in the examples\n"
        f"3. How to score it on a 1-5 scale\n\n"
        f"Format each line as: feature_key | why_important | scoring_scale\n\n"
        f"Examples:\n"
        f"visual_quality | Top-performing ads show polished, professional visuals vs lower performers have amateur graphics | 1=poor/amateur, 5=premium/polished\n"
        f"brand_fit | High performers clearly align with brand positioning vs poor performers feel off-brand | 1=completely off-brand, 5=perfectly aligned\n\n"
        f"Generate {num_features} features based on what you observe in the data:"
    )
    
    user_content = [{"type": "text", "text": user_prompt_text}]
    if brand_samples_content:
        user_content.extend(brand_samples_content)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0
    }
    
    samples_info = f" + {len(brand_performance_samples)} brand samples" if brand_performance_samples else ""
    insight_info = "with similar brand insights" if cross_brand_reasoning else ""
    print(f"[Stage 2] Generating features from few-shot examples (fewshot_groups={len(fewshot_examples)}{samples_info}) {insight_info}...")
    
    resp = requests.post(API_URL_OPENAI, headers=HEADERS_OPENAI, json=payload, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI Stage 2 API error {resp.status_code}: {resp.text[:500]}")
    
    content = resp.json()["choices"][0]["message"]["content"].strip()
    
    selected = {}
    for line in content.split("\n"):
        line = line.strip()
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            feature_key = parts[0].lower().replace(" ", "_").replace("-", "_")
            description = parts[1] if len(parts) > 1 else "Feature based on ranking patterns"
            selected[feature_key] = description
    
    if selected:
        print(f"[Stage 2] ✓ Generated {len(selected)} features for brand {brand_id} from few-shot patterns:")
        for k, desc in selected.items():
            print(f"  - {k}: {desc}...")
    else:
        print(f"[Stage 2] WARNING: No features generated. Using defaults.")
    return selected

def build_fewshot_examples_multimetric(df: pd.DataFrame, metric_modes: List[str], max_examples: int = FEWSHOT_MAX_EXAMPLES, example_size: int = FEWSHOT_EXAMPLE_SIZE, target_metric: str = None) -> List[Dict[str, Any]]:

    if not metric_modes:
        return []

    target_cols = [METRIC_MODES[m]["col"] for m in metric_modes]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        print(f"[Few-shot] Missing columns: {missing}")
        return []

    df_valid = df.copy()
    for col in target_cols:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")
    df_valid = df_valid.dropna(subset=target_cols)

    if df_valid.empty:
        return []

    ranking_mode = target_metric if target_metric and target_metric in metric_modes else metric_modes[0]
    ranking_col = METRIC_MODES[ranking_mode]["col"]
    ranking_direction = METRIC_MODES[ranking_mode]["direction"]
    ascending = True if ranking_direction == "lower" else False
    
    df_sorted = df_valid.sort_values(ranking_col, ascending=ascending).reset_index(drop=True)
    examples: List[Dict[str, Any]] = []

    total_needed = max_examples * example_size
    df_top = df_sorted.head(total_needed)

    for i in range(max_examples):
        start = i * example_size
        end = start + example_size
        chunk = df_top.iloc[start:end]
        if chunk.empty:
            break

        ads = []
        ranking_order = []
        metric_values = {} 
        
        for _, r in chunk.iterrows():
            mm = row_to_mm_dict(r)
            ads.append(mm)
            ad_id = mm["ad_id"]
            ranking_order.append(ad_id)
            
            metric_values[ad_id] = {}
            for mode in metric_modes:
                col = METRIC_MODES[mode]["col"]
                metric_values[ad_id][mode] = float(r[col])

        examples.append(
            {"example_id": i + 1,
                "ads": ads,
                "ranking_order": ranking_order,
                "metric_values": metric_values,
                "primary_metric": ranking_mode, 
            }
        )

    return examples

# =======================
# Self-Critique and Refinement for Scoring
# =======================
def parse_reasoning_and_scores(content: str, feature_keys: List[str] = None) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:

    reasoning_dict: Dict[str, str] = {}
    scores_dict: Dict[str, Dict[str, float]] = {}
    
    if not content or not feature_keys:
        return reasoning_dict, scores_dict
    
    # Extract reasoning before scores block
    lines = content.strip().splitlines()
    scores_start_idx = None
    for i, line in enumerate(lines):
        if '<SCORES_START>' in line.upper():
            scores_start_idx = i
            break
    
    # Parse reasoning section (before scores)
    if scores_start_idx is not None and scores_start_idx > 0:
        reasoning_section = '\n'.join(lines[:scores_start_idx])
        # Simple parsing: look for "Ad {ad_id}:" or "ad_id {ad_id}" patterns
        import re
        # Match patterns like "Ad 123:" or "ad_id: 123" followed by text
        pattern = r'(?:Ad|ad_id)[:\s]+([\w-]+)[:\s]*([^\n]+(?:\n(?!(?:Ad|ad_id)[:\s]+)[^\n]+)*)'
        matches = re.finditer(pattern, reasoning_section, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            ad_id = match.group(1).strip()
            reasoning = match.group(2).strip()
            reasoning_dict[ad_id] = reasoning
    
    # Parse scores using existing logic
    scores_dict = parse_line_based_scores(content, feature_keys)
    
    return reasoning_dict, scores_dict

def parse_line_based_scores(content: str, feature_keys: List[str] = None) -> Dict[str, Dict[str, float]]:

    scores: Dict[str, Dict[str, float]] = {}
    if not content:
        return scores
    
    if not feature_keys:
        feature_keys = []
    
    num_features = len(feature_keys)
    if num_features == 0:
        return scores
    
    # Extract content between tags if present
    lines_to_parse = content.strip().splitlines()
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines_to_parse):
        if "<SCORES_START>" in line:
            start_idx = i + 1
        elif "<SCORES_END>" in line:
            end_idx = i
            break
    
    if start_idx is not None and end_idx is not None:
        lines_to_parse = lines_to_parse[start_idx:end_idx]
    elif start_idx is not None:
        lines_to_parse = lines_to_parse[start_idx:]
    
    # Parse each line
    parsed_count = 0
    for line in lines_to_parse:
        line = line.strip()
        
        # Skip empty lines, comments, headers, separators
        if not line:
            continue
        if line.startswith("#") or line.startswith("=") or line.startswith("-"):
            continue
        if line.lower().startswith("ad_id") or line.lower() == "ad_id":
            continue
        if any(word in line.lower() for word in ["initial scores", "critique scores", "final scores", "scores table"]):
            continue
        
        parts = line.split()
        if len(parts) < num_features + 1:
            continue
        ad_id = parts[0]
        score_values = []
        for i in range(1, min(len(parts), num_features + 1)):
            score_str = parts[i].strip()
            try:
                score_val = float(score_str)
                if score_val < 0.5 or score_val > 5.5:
                    break
                score_values.append(score_val)
            except ValueError:
                break
        
        if len(score_values) == num_features:
            scores[ad_id] = {feature_keys[i]: score_values[i] for i in range(num_features)}
            parsed_count += 1
    return scores

# ---------- Critique initial scores ----------
def build_gpt_critique_system_msg(feature_keys: List[str]) -> str:
    header = "ad_id " + " ".join(feature_keys)
    return (
        "You are the second-stage critic. You receive initial reasoning and 1-5 scores for each ad across multiple features.\n"
        "Your job: Review the initial scorer's reasoning and detect inconsistencies, scale collapse, bias, or missing penalties.\n\n"
        "Rules:\n"
        "- First, for each ad, provide your critique reasoning (1-2 sentences explaining what the initial scorer got right or wrong).\n"
        "- Then suggest corrected scores (1-5 integers).\n"
        "- Keep the SAME features and scale (1-5 integers).\n"
        "- If initial scoring looks reasonable, keep the same score but still explain why.\n"
        "- If you adjust, stay within 1-5 and avoid inflating everything.\n"
        "- Be strict on: clarity of message, brand fit, misleading visuals, and over-claiming captions.\n\n"
        "Format your response as:\n"
        "Ad {ad_id}: [Your critique reasoning]\n"
        "...\n\n"
        "Then output the scores block with one blank line before it:\n"
        f"<SCORES_START>\n{header}\n123 4 3 5\n<...>\n<SCORES_END>"
    )

def build_gpt_critique_messages(
    mm_items: List[Dict[str, Any]],
    initial_scores: Dict[str, Dict[str, float]],
    initial_reasoning: Dict[str, str],
    feature_keys: List[str],
) -> List[Dict[str, Any]]:
    sys = build_gpt_critique_system_msg(feature_keys)
    messages = [{"role": "system", "content": [{"type": "text", "text": sys}]}]

    # Provide ad context with initial reasoning and scores
    ctx_lines = ["Initial scorer's reasoning and scores:"]
    ctx_lines.append("")
    
    # Add reasoning for each ad
    for ad_id in initial_scores.keys():
        reasoning = initial_reasoning.get(ad_id, "No reasoning provided")
        ctx_lines.append(f"Ad {ad_id}: {reasoning}")
    
    ctx_lines.append("")
    ctx_lines.append("Initial scores table (ad_id then features):")
    header = "ad_id " + " ".join(feature_keys)
    ctx_lines.append(header)
    for ad_id in initial_scores.keys():
        row = [ad_id] + [str(initial_scores[ad_id].get(k, 0)) for k in feature_keys]
        ctx_lines.append(" ".join(row))

    content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(ctx_lines)}]

    content.append({"type": "text", "text": "\nAd contexts:"})
    for i, a in enumerate(mm_items, 1):
        content.append({
            "type": "text",
            "text": json.dumps({"ad_id": a.get("ad_id"), "caption": a.get("caption", "")}, ensure_ascii=False),
        })
        if a.get("image_data_url"):
            content.append({"type": "image_url", "image_url": {"url": a["image_data_url"]}})

    messages.append({"role": "user", "content": content})
    return messages

def call_gpt_critique_scores(
    mm_items: List[Dict[str, Any]],
    initial_scores: Dict[str, Dict[str, float]],
    initial_reasoning: Dict[str, str],
    feature_keys: List[str],
    temperature: float = 0.0,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]], str]:
    import time
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            messages = build_gpt_critique_messages(mm_items, initial_scores, initial_reasoning, feature_keys)
            payload = {"model": OPENAI_MODEL_NAME, "messages": messages, "temperature": temperature}
            resp = requests.post(API_URL_OPENAI, headers=HEADERS_OPENAI, json=payload, timeout=180)
            
            if resp.status_code == 500 and attempt < max_retries - 1:
                print(f"    [Retry {attempt+1}/{max_retries}] Server error 500, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI critique API error {resp.status_code}: {resp.text[:500]}")

            content = resp.json()["choices"][0]["message"]["content"].strip()
            reasoning, scores = parse_reasoning_and_scores(content, feature_keys=feature_keys)
            return reasoning, scores, content
        except Exception as e:
            if attempt < max_retries - 1 and (resp.status_code == 500 or "500" in str(e)):
                print(f"    [Retry {attempt+1}/{max_retries}] Error, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                raise
    reasoning, scores = parse_reasoning_and_scores("", feature_keys=feature_keys)
    
    if not scores:
        print(f" Critique stage: Failed to parse any scores from {len(mm_items)} ads")
        print(f"Raw response preview: {content[:300]}...")
    
    return reasoning, scores, content

# ---------- Final Refinement ----------
def build_gpt_final_system_msg(feature_keys: List[str]) -> str:
    header = "ad_id " + " ".join(feature_keys)
    return (
        "You are the third-stage arbiter. You see both initial scorer's scores and critic's reasoning.\n"
        "Decide the FINAL scores (1-5 integers) for each ad_id, feature-wise.\n"
        "Rules:\n"
        "- First, for each ad, provide your final reasoning (1-2 sentences explaining your decision).\n"
        "- Prefer critic adjustments when they fix scale compression, bias, or obvious errors.\n"
        "- If critic over-corrects or seems inconsistent with evidence, keep the initial value.\n"
        "- Preserve full-scale usage; avoid all ads ending 4-5.\n\n"
        "Format your response as:\n"
        "Ad {ad_id}: [Your final reasoning]\n"
        "...\n\n"
        "Then output the final scores block:\n"
        f"<SCORES_START>\n{header}\n123 4 3 5\n<...>\n<SCORES_END>"
    )

def build_gpt_final_messages(
    mm_items: List[Dict[str, Any]],
    initial_scores: Dict[str, Dict[str, float]],
    initial_reasoning: Dict[str, str],
    critique_scores: Dict[str, Dict[str, float]],
    critique_reasoning: Dict[str, str],
    feature_keys: List[str],
) -> List[Dict[str, Any]]:
    sys = build_gpt_final_system_msg(feature_keys)
    messages = [{"role": "system", "content": [{"type": "text", "text": sys}]}]

    def table_from_scores_and_reasoning(tag: str, scores: Dict[str, Dict[str, float]], reasoning: Dict[str, str]) -> List[str]:
        lines = [f"{tag}"]
        lines.append("Reasoning:")
        for ad_id in scores.keys():
            reason = reasoning.get(ad_id, "No reasoning provided")
            lines.append(f"  Ad {ad_id}: {reason}")
        lines.append("")
        lines.append("Scores:")
        lines.append("ad_id " + " ".join(feature_keys))
        for ad_id in scores.keys():
            row = [ad_id] + [str(scores[ad_id].get(k, 0)) for k in feature_keys]
            lines.append(" ".join(row))
        return lines

    lines = table_from_scores_and_reasoning("Critique scorer", initial_scores, critique_reasoning)

    content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    content.append({"type": "text", "text": "\nAd contexts:"})
    for i, a in enumerate(mm_items, 1):
        content.append({
            "type": "text",
            "text": json.dumps({"ad_id": a.get("ad_id"), "caption": a.get("caption", "")}, ensure_ascii=False),
        })
        if a.get("image_data_url"):
            content.append({"type": "image_url", "image_url": {"url": a["image_data_url"]}})

    messages.append({"role": "user", "content": content})
    return messages

def call_gpt_final_scores(
    mm_items: List[Dict[str, Any]],
    initial_scores: Dict[str, Dict[str, float]],
    initial_reasoning: Dict[str, str],
    critique_scores: Dict[str, Dict[str, float]],
    critique_reasoning: Dict[str, str],
    feature_keys: List[str],
    temperature: float = 0.0,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]], str]:
    import time
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            messages = build_gpt_final_messages(mm_items, initial_scores, initial_reasoning, critique_scores, critique_reasoning, feature_keys)
            payload = {"model": OPENAI_MODEL_NAME, "messages": messages, "temperature": temperature}
            resp = requests.post(API_URL_OPENAI, headers=HEADERS_OPENAI, json=payload, timeout=180)
            
            if resp.status_code == 500 and attempt < max_retries - 1:
                print(f"    [Retry {attempt+1}/{max_retries}] Server error 500, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI final API error {resp.status_code}: {resp.text[:500]}")

            content = resp.json()["choices"][0]["message"]["content"].strip()
            reasoning, scores = parse_reasoning_and_scores(content, feature_keys=feature_keys)
            return reasoning, scores, content
        except Exception as e:
            if attempt < max_retries - 1 and (resp.status_code == 500 or "500" in str(e)):
                print(f"    [Retry {attempt+1}/{max_retries}] Error, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                raise
    
    # Fallback
    reasoning, scores = parse_reasoning_and_scores("", feature_keys=feature_keys)
    return reasoning, scores, ""

# ---------- Initial Scoring ----------
def build_gpt_initial_system_msg(
    brand_id: str, 
    metric_modes: List[str] = None,
    selected_features: Dict[str, str] = None,
    cross_brand_reasoning: str = None,
) -> str:
  
    # 메트릭 정보 설명 (더 자세한 설명)
    metric_info = ""
    if metric_modes:
        metric_lines = []
        for mode in metric_modes:
            col = METRIC_MODES[mode]["col"]
            direction = METRIC_MODES[mode]["direction"]
            desc = {
                "spc": "Spend Per Click - Lower is BETTER (cost efficiency: how much spent per click)",
                "ctr": "Click-Through Rate - Higher is BETTER (engagement: click rate per impression)",
                "spi": "Spend Per Impression - Lower is BETTER (cost efficiency: how much spent per impression)"
            }.get(mode, mode)
            metric_lines.append(f"  - {mode.upper()} ({col}): {desc}")
        metric_info = (
            "Performance Metrics Used in Examples:\n"
            + "\n".join(metric_lines) + "\n"
            + f"Primary Ranking Metric: {metric_modes[0].upper()} (but all metrics are provided for analysis)\n\n"
        )
    
    feature_info = ""
    feature_order = []
    if selected_features:
        feature_lines = []
        feature_order = list(selected_features.keys())
        for feat_key, feat_desc in selected_features.items():
            feature_lines.append(f"  - {feat_key}: {feat_desc}")
        feature_info = (
            "Scoring dimensions for this brand (use integers from 1 to 5):\n"
            + "\n".join(feature_lines) + "\n\n"
        )
        
    return (
        "You are the first-stage evaluator for Instagram ads.\n"
        + metric_info
        + "You will be given:\n"
          "- A small number of training examples from this brand, where ads are ordered by performance.\n"
          "- A new set of ads (caption + image) to evaluate.\n\n"
        + "For EACH new ad_id, do the following:\n"
        "  1) Briefly explain your reasoning (2–3 sentences) about the ad, based on BOTH the caption and the image.\n"
        "  2) Then, at the END of the entire response, output a final scores block.\n\n"
        + feature_info
        + "Scale usage guidelines (VERY IMPORTANT to avoid bias):\n"
        "- Use the FULL 1–5 scale. Do NOT collapse everything into 3–5.\n"
        "- 1 = very poor / unacceptable\n"
        "- 2 = below average / needs improvement\n"
        "- 3 = acceptable / average\n"
        "- 4 = good / above average\n"
        "- 5 = excellent / outstanding\n\n"
        "CRITICAL: FINAL SCORES BLOCK FORMAT (very important - parse will fail if not exact):\n"
        f"Output exactly like this at the very end, with ONE BLANK LINE before it:\n\n"
        f"<SCORES_START>\n"
        f"ad_id {' '.join(feature_order)}\n"
        f"123456789 4 5 4\n"
        f"987654321 3 4 5\n"
        f"<SCORES_END>\n\n"
        "NOTES:\n"
        "- Each line in the block has ad_id followed by scores separated by spaces\n"
        "- Scores must be integers 1-5\n"
        "- One ad per line\n"
        "- Do not include any other text in the scores block\n"
    )

def build_gpt_initial_messages(
    brand_id: str,
    mm_items: List[Dict[str, Any]],
    fewshot_examples: List[Dict[str, Any]] = None,
    metric_modes: List[str] = None,
    selected_features: Dict[str, str] = None,
    cross_brand_reasoning: str = None,
) -> List[Dict[str, Any]]:
    sys = build_gpt_initial_system_msg(brand_id, metric_modes, selected_features, cross_brand_reasoning)
    messages = [
        {"role": "system",
        "content": [{"type": "text", "text": sys}]}
    ]

    content: List[Dict[str, Any]] = []

    if fewshot_examples:
        primary_metric = fewshot_examples[0].get("primary_metric", "") if fewshot_examples else ""
        primary_col = METRIC_MODES.get(primary_metric, {}).get("col", "")
        primary_direction = METRIC_MODES.get(primary_metric, {}).get("direction", "higher")
        
        intro_fs = [
            "FEW-SHOT TRAINING EXAMPLES FROM THIS BRAND",
            "="*60,
            f"Ads below are RANKED from BEST to WORST based on {primary_metric.upper()} ({primary_col}).",
            f"(Note: {primary_direction} {primary_col} = better performance)",
            f"\nYou can see all metrics for each ad to understand tradeoffs between metrics.",
            "",
        ]
        content.append({"type": "text", "text": "\n".join(intro_fs)})

        for ex in fewshot_examples:
            ex_id = ex["example_id"]
            ranking_order_text = " > ".join([f"Ad{aid}" for aid in ex["ranking_order"]])
            content.append(
                {"type": "text", "text": f"\n[Example Group {ex_id}] Ranking (left=BEST, right=WORST): {ranking_order_text}\n"}
            )
            
            for rank_idx, ad in enumerate(ex["ads"], 1):
                ad_id = ad["ad_id"]
                
                total_ads = len(ex["ads"])
                if rank_idx <= total_ads // 3 + 1:
                    tier = "[HIGH PERFORMER]"
                elif rank_idx <= 2 * total_ads // 3:
                    tier = "[MEDIUM PERFORMER]"
                else:
                    tier = "[LOW PERFORMER]"
                
                metric_values = ex["metric_values"].get(ad_id, {})
                metrics_lines = []
                for mode in (metric_modes or []):
                    col = METRIC_MODES[mode]["col"]
                    val = metric_values.get(mode, 0)
                    metrics_lines.append(f"{mode.upper()}({col})={val:.4f}")
                metrics_str = " | ".join(metrics_lines)

                content.append(
                    {
                        "type": "text",
                        "text": f"  Rank {rank_idx} {tier}\n    Ad {ad_id}: {metrics_str}\n    Caption: {ad['caption'][:100]}...",
                    }
                )
                
                if ad.get("image_data_url"):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": ad["image_data_url"]},
                        }
                    )
            
    
    intro_text = [
        "",
        "Now evaluate the following ads. For each ad you should reason briefly,",
        "and then at the end output a single scores block for ALL of these ads.",
        "",
        "=== ADS TO SCORE ===",
    ]
    content.append({"type": "text", "text": "\n".join(intro_text)})

    for i, a in enumerate(mm_items, 1):
        content.append({"type": "text", "text": f"\n--- AD {i} ---"})
        content.append({
            "type": "text",
            "text": json.dumps(
                {
                    "ad_id": a["ad_id"],
                    "caption": a["caption"],
                },
                ensure_ascii=False,
            ),
        })
        if a.get("image_data_url"):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": a["image_data_url"]},
                }
            )

    messages.append({"role": "user", "content": content})
    return messages

def call_gpt_initial_scores(
    brand_id: str, mm_items: List[Dict[str, Any]], temperature: float = 0.0,
    fewshot_examples: List[Dict[str, Any]] = None, metric_modes: List[str] = None,
    selected_features: Dict[str, str] = None, cross_brand_reasoning: str = None) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]], str]:

    messages = build_gpt_initial_messages(
        brand_id,
        mm_items,
        fewshot_examples=fewshot_examples,
        metric_modes=metric_modes,
        selected_features=selected_features,
        cross_brand_reasoning=cross_brand_reasoning,
    )
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": messages,
        "temperature": temperature
    }
    resp = requests.post(API_URL_OPENAI, headers=HEADERS_OPENAI, json=payload, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI initial API error {resp.status_code}: {resp.text[:500]}")

    content = resp.json()["choices"][0]["message"]["content"].strip()
    feature_keys = list(selected_features.keys())
    reasoning, scores = parse_reasoning_and_scores(content, feature_keys=feature_keys)
    if not scores:
        log_path = f"gpt_initial_raw_{brand_id}.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f" GPT initial: Failed to parse any scores. Raw content saved to: {log_path}")
        except Exception:
            print(" GPT initial: Failed to parse any scores (and could not save raw content).")

    return reasoning, scores, content

def augment_with_llm_features_multimodel(
    brand_id: str,
    df: pd.DataFrame,
    fewshot_examples: List[Dict[str, Any]] = None,
    metric_modes: List[str] = None,
    selected_features: Dict[str, str] = None,
    cross_brand_reasoning: str = None,
    use_critique: bool = True,
) -> pd.DataFrame:

    if df is None or df.empty:
        if selected_features:
            df2 = df.copy()
            for col in selected_features.keys():
                df2[f"llm_{col}"] = 0.0
            return df2
        return df

    if not selected_features:
        selected_features = {}

    mm_items_all = [row_to_mm_dict(r) for _, r in df.iterrows()]
    ad_ids_all = [m["ad_id"] for m in mm_items_all]

    final_scores_all: Dict[str, Dict[str, float]] = {}

    scoring_mode_str = "3-stage critique" if use_critique else "single-pass"
    cross_brand_str = " (WITH cross-brand reasoning)" if cross_brand_reasoning else " (without cross-brand reasoning)"
    print(f"[{brand_id}] Stage 3: {scoring_mode_str} LLM scoring{cross_brand_str} - {len(selected_features)} features...")
    
    n = len(mm_items_all)
    for start in range(0, n, LLM_BATCH_SIZE):
        end = min(start + LLM_BATCH_SIZE, n)
        batch_items = mm_items_all[start:end]
        feature_keys = list(selected_features.keys())

        print(f"  [{brand_id}] LLM batch {start}–{end} / {n}")
        
        if use_critique:
            # 3-stage scoring: initial -> critique -> final
            try:
                init_reasoning, init_scores, raw_init = call_gpt_initial_scores(
                    brand_id,
                    batch_items,
                    temperature=0.0,
                    fewshot_examples=fewshot_examples,
                    metric_modes=metric_modes,
                    selected_features=selected_features,
                    cross_brand_reasoning=cross_brand_reasoning,
                )
            except RuntimeError as e:
                print(f"   [{brand_id}] Initial stage failed {start}–{end}: {e}")
                init_reasoning, init_scores = {}, {}

            if not init_scores:
                print(f"   [{brand_id}] No initial scores parsed for batch {start}–{end}; skipping critique/final.")
                continue

            try:
                critique_reasoning, critique_scores, raw_crit = call_gpt_critique_scores(
                    batch_items,
                    init_scores,
                    init_reasoning,
                    feature_keys,
                    temperature=0.0,
                )
            except RuntimeError as e:
                print(f"   [{brand_id}] Critique stage failed {start}–{end}: {e}")
                critique_reasoning, critique_scores = {}, {}

            # Fallback to initial if critique missing
            critique_scores = critique_scores if critique_scores else init_scores
            critique_reasoning = critique_reasoning if critique_reasoning else init_reasoning

            try:
                final_reasoning, final_scores, raw_final = call_gpt_final_scores(
                    batch_items,
                    init_scores,
                    init_reasoning,
                    critique_scores,
                    critique_reasoning,
                    feature_keys,
                    temperature=0.0,
                )
            except RuntimeError as e:
                print(f"   [{brand_id}] Final stage failed {start}–{end}: {e}. Using critique scores.")
                final_reasoning, final_scores = critique_reasoning, critique_scores

            for ad_id, v in final_scores.items():
                final_scores_all[ad_id] = v
        else:
            # Single-pass scoring: direct inference
            try:
                single_reasoning, single_scores, raw_single = call_gpt_initial_scores(
                    brand_id,
                    batch_items,
                    temperature=0.0,
                    fewshot_examples=fewshot_examples,
                    metric_modes=metric_modes,
                    selected_features=selected_features,
                    cross_brand_reasoning=cross_brand_reasoning,
                )
            except RuntimeError as e:
                print(f"   [{brand_id}] Single-pass scoring failed {start}–{end}: {e}")
                single_reasoning, single_scores = {}, {}

            if not single_scores:
                print(f"   [{brand_id}] No scores parsed for batch {start}–{end}")
                continue

            for ad_id, v in single_scores.items():
                final_scores_all[ad_id] = v

    rows = []
    for ad_id in ad_ids_all:
        v = final_scores_all.get(ad_id, {})
        row = {"ad_id": ad_id}
        for feat_key in selected_features.keys():
            col_name = f"llm_{feat_key}"
            row[col_name] = float(v.get(feat_key, 0.0))
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_df["ad_id"] = feat_df["ad_id"].astype(str)

    df2 = df.copy()
    df2["ad_id"] = df2["ad_id"].astype(str)
    df2 = df2.merge(feat_df, on="ad_id", how="left")

    for feat_key in selected_features.keys():
        col_name = f"llm_{feat_key}"
        if col_name not in df2.columns:
            df2[col_name] = 0.0
        df2[col_name] = pd.to_numeric(df2[col_name], errors="coerce").fillna(0.0)

    return df2

# =======================
# 4) LightGBM Ranker
# =======================
def build_tabular_features(df: pd.DataFrame, llm_feature_cols: List[str] = None) -> pd.DataFrame:
    if llm_feature_cols is None:
        llm_feature_cols = []
    
    out = pd.DataFrame(
        {
            "ad_id": df["ad_id"].astype(str),
        }
    )
    for c in TABULAR_COLS_NUM:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    
    for c in llm_feature_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0

    tmp = out.copy()

    reserved = set(["ad_id"] + TABULAR_COLS_NUM + llm_feature_cols)
    auto_cat_cols: List[str] = []
    for c in df.columns:
        if c in reserved:
            continue
        if df[c].dtype == object or str(df[c].dtype).startswith("category"):
            if df[c].nunique(dropna=True) <= 60:
                auto_cat_cols.append(c)

    cat_cols = list(dict.fromkeys(TABULAR_COLS_CAT + auto_cat_cols))

    for c in cat_cols:
        if c in df.columns:
            tmp[c] = df[c].astype(str).fillna("NA")
        else:
            tmp[c] = "NA"

    out = pd.get_dummies(tmp, columns=cat_cols, dummy_na=False)
    out.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', str(col)) for col in out.columns]
    
    return out

# =======================
# 5) LLM Direct Ranking
# =======================
def llm_direct_ranking(
    brand_id: str,
    df: pd.DataFrame,
    target_col: str,
    direction: str,
    selected_features: Dict[str, str],
    llm_feature_cols: List[str],
    cross_brand_reasoning: str = None,
    batch_size: int = 10,
) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame(columns=["ad_id", "score"])
    
    print(f"[{brand_id}] Starting LLM Direct Ranking for {len(df)} ads...")
    
    # Prepare multimodal items
    mm_items_all = []
    for _, row in df.iterrows():
        mm_dict = row_to_mm_dict(row)
        llm_scores = {}
        for feat_col in llm_feature_cols:
            if feat_col in row:
                feat_name = feat_col.replace("llm_", "")
                llm_scores[feat_name] = float(row[feat_col])
        mm_dict["llm_feature_scores"] = llm_scores
        
        tabular_info = {}
        for col in TABULAR_COLS_NUM:
            if col in row:
                tabular_info[col] = float(row[col])
        for col in TABULAR_COLS_CAT:
            if col in row:
                tabular_info[col] = str(row[col])
        mm_dict["tabular_features"] = tabular_info
        mm_items_all.append(mm_dict)    
    all_scores = {}
    
    n = len(mm_items_all)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_items = mm_items_all[start:end]
        
        print(f"  [{brand_id}] Processing batch {start+1}-{end}/{n}...")
        
        try:
            batch_scores = call_llm_direct_ranking(
                brand_id=brand_id,
                mm_items=batch_items,
                target_col=target_col,
                direction=direction,
                selected_features=selected_features,
                cross_brand_reasoning=cross_brand_reasoning,
            )
            all_scores.update(batch_scores)
        except Exception as e:
            print(f"   [{brand_id}] Batch {start}-{end} failed: {e}")
            for item in batch_items:
                all_scores[item["ad_id"]] = 0.0
    
    results = []
    for ad_id in [item["ad_id"] for item in mm_items_all]:
        results.append({
            "ad_id": str(ad_id),
            "score": all_scores.get(ad_id, 0.5)
        })
    
    print(f"[{brand_id}] LLM Direct Ranking completed: {len(results)} ads scored")
    return pd.DataFrame(results)

def call_llm_direct_ranking(
    brand_id: str,
    mm_items: List[Dict[str, Any]],
    target_col: str,
    direction: str,
    selected_features: Dict[str, str],
    cross_brand_reasoning: str = None,
) -> Dict[str, float]:

    # Build system message
    sys_msg = build_llm_ranking_system_msg(
        target_col=target_col,
        direction=direction,
        selected_features=selected_features,
        cross_brand_reasoning=cross_brand_reasoning,
    )
    
    # Build user message with all ad contexts
    user_content = build_llm_ranking_user_content(mm_items)
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        {"role": "user", "content": user_content},
    ]
    
    payload = {
        "model": OPENAI_MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
    }
    
    resp = requests.post(API_URL_OPENAI, headers=HEADERS_OPENAI, json=payload, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI LLM ranking API error {resp.status_code}: {resp.text[:500]}")
    
    content = resp.json()["choices"][0]["message"]["content"].strip()
    
    # Parse scores from response
    scores = parse_llm_ranking_scores(content, mm_items)
    return scores

def build_llm_ranking_system_msg(
    target_col: str,
    direction: str,
    selected_features: Dict[str, str],
    cross_brand_reasoning: str = None,
) -> str:
    direction_text = "HIGHER is BETTER" if direction == "higher" else "LOWER is BETTER"
    
    features_text = "\n".join([f"  - {k}: {v}" for k, v in selected_features.items()])
    
    return (
        f"You are an expert ad performance evaluator. Your task is to predict which ads will perform best on the metric: {target_col} ({direction_text}).\n\n"
        f"For each ad, you will receive:\n"
        f"1. Image and caption\n"
        f"2. LLM-generated feature scores (1-5) based on these criteria:\n{features_text}\n"
        f"Your job:\n"
        f"- Analyze all provided information holistically\n"
        f"- Consider how the image, caption, and feature scores combine to predict {target_col} performance\n"
        f"- Output a ranking score (0.0 to 1.0) for each ad, where HIGHER score = BETTER expected {target_col} performance\n"
        f"- Use the full range 0.0-1.0; avoid clustering scores\n\n"
        f"Output format:\n"
        f"<RANKING_START>\n"
        f"ad_id score\n"
        f"123 0.85\n"
        f"456 0.42\n"
        f"<RANKING_END>"
    )

def build_llm_ranking_user_content(mm_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content = [{"type": "text", "text": "Ads to rank:"}]
    
    for i, item in enumerate(mm_items, 1):
        ad_id = item["ad_id"]
        caption = item.get("caption", "")
        llm_scores = item.get("llm_feature_scores", {})
        tabular = item.get("tabular_features", {})
        
        # Format LLM feature scores
        llm_scores_text = ", ".join([f"{k}={v:.1f}" for k, v in llm_scores.items()])
        
        # Format tabular features
        tabular_text = ", ".join([f"{k}={v}" for k, v in tabular.items()])
        
        content.append({
            "type": "text",
            "text": (
                f"\n--- Ad {i} (ID: {ad_id}) ---\n"
                f"Caption: {caption[:150]}...\n"
                f"LLM Features: {llm_scores_text}\n"
                f"Tabular: {tabular_text}"
            )
        })
        
        if item.get("image_data_url"):
            content.append({
                "type": "image_url",
                "image_url": {"url": item["image_data_url"]}
            })
    
    return content

def parse_llm_ranking_scores(content: str, mm_items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Parse ranking scores from LLM response.
    Expected format:
    <RANKING_START>
    ad_id score
    123 0.85
    456 0.42
    <RANKING_END>
    """
    scores = {}
    
    lines = content.strip().splitlines()
    in_ranking = False
    
    for line in lines:
        line = line.strip()
        if "<RANKING_START>" in line.upper():
            in_ranking = True
            continue
        if "<RANKING_END>" in line.upper():
            in_ranking = False
            break
        
        if not in_ranking or not line:
            continue
        
        # Skip header
        if line.lower().startswith("ad_id"):
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            ad_id = parts[0]
            try:
                score = float(parts[1])
                scores[ad_id] = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except ValueError:
                continue
    
    # Fill missing scores with default
    for item in mm_items:
        ad_id = item["ad_id"]
        if ad_id not in scores:
            scores[ad_id] = 0.5
    
    return scores

def train_lgbm_ranker(train_df: pd.DataFrame, target_col: str, direction: str = "higher", llm_feature_cols: List[str] = None, tune_hyperparams: bool = False):
    if llm_feature_cols is None:
        llm_feature_cols = []
    
    feats = build_tabular_features(train_df, llm_feature_cols=llm_feature_cols)
    y = train_df[["ad_id", target_col]].copy()
    y["ad_id"] = y["ad_id"].astype(str)
    y["relevance"] = make_relevance_grades(y[target_col], n_bins=5, direction=direction)

    dtrain = feats.merge(y[["ad_id", "relevance", target_col]], on="ad_id", how="inner")

    group = [len(dtrain)]
    feature_cols = [c for c in dtrain.columns if c not in ["ad_id", target_col, "relevance"]]

    train_set = lgb.Dataset(dtrain[feature_cols], label=dtrain["relevance"], group=group, free_raw_data=False)
    label_gain = [0, 1, 3, 7, 15][: (dtrain["relevance"].max() + 1)]

    params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5],
            "learning_rate": 0.05,
            "num_leaves": 10,
            "is_higher_better": True,
            "min_data_in_leaf": 5,
            "feature_pre_filter": False,
            "verbosity": -1,
            "seed": SEED,
            "label_gain": label_gain,
        }
    
    model = lgb.train(params, train_set, num_boost_round=300)
    return model, feature_cols

def predict_lgbm_scores(model, feature_cols: list, df: pd.DataFrame, llm_feature_cols: List[str] = None) -> pd.DataFrame:
    if llm_feature_cols is None:
        llm_feature_cols = []
    
    feats = build_tabular_features(df, llm_feature_cols=llm_feature_cols)
    ad_ids = feats["ad_id"].astype(str).values

    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = 0
    feats = feats.reindex(columns=feature_cols, fill_value=0)

    preds = model.predict(feats)
    return pd.DataFrame({"ad_id": ad_ids, "score": preds})

class MLPHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class LinearHead(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

@torch.no_grad()
def predict_torch_ranker(head, df: pd.DataFrame, feature_cols: List[str], llm_feature_cols: List[str], device="cuda", stats: Dict[str, Any] = None):
    feats = build_tabular_features(df, llm_feature_cols=llm_feature_cols)
    ad_ids = feats["ad_id"].astype(str).values

    for col in feature_cols:
        if col not in feats.columns:
            feats[col] = 0
    feats = feats.reindex(columns=feature_cols, fill_value=0)
    
    X = feats.values.astype(np.float32)
    X_t = torch.tensor(X, device=device)
    head.eval()
    scores = head(X_t).view(-1).detach().cpu().numpy()
    
    return pd.DataFrame({"ad_id": ad_ids, "score": scores})

def pairwise_logistic_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = scores.shape[0]
    if m <= 1:
        return scores.new_tensor(0.0)

    # create pair mask
    yi = y.view(m, 1)
    yj = y.view(1, m)
    mask = (yi > yj)  # (m,m)

    if mask.sum() == 0:
        return scores.new_tensor(0.0)

    si = scores.view(m, 1)
    sj = scores.view(1, m)
    sdiff = si - sj  # (m,m)

    loss = F.softplus(-sdiff)[mask].mean()
    return loss

def train_pairwise_ranker_head(train_df: pd.DataFrame, target_col: str, llm_feature_cols: List[str],  head_type: str = "mlp", direction: str = "higher",
    n_bins: int = 5, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 50, device: str = "cuda"):
    feats = build_tabular_features(train_df, llm_feature_cols=llm_feature_cols).copy()
    feats["ad_id"] = feats["ad_id"].astype(str)
    y_df = train_df[["ad_id", target_col]].copy()
    y_df["ad_id"] = y_df["ad_id"].astype(str)
    y_df["relevance"] = make_relevance_grades(y_df[target_col], n_bins=n_bins, direction=direction)

    dtrain = feats.merge(y_df[["ad_id", "relevance"]], on="ad_id", how="inner")
    feature_cols = [c for c in dtrain.columns if c not in ["ad_id", "relevance"]]
    X = dtrain[feature_cols].values.astype(np.float32)
    rel = dtrain["relevance"].values.astype(np.int64)

    X_t = torch.tensor(X, device=device)
    rel_t = torch.tensor(rel, device=device, dtype=torch.long)

    in_dim = X_t.shape[1]
    if head_type == "linear":
        head = LinearHead(in_dim=in_dim)
    elif head_type == "mlp":
        head = MLPHead(in_dim=in_dim)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    idx = torch.arange(len(dtrain), device=device, dtype=torch.long)

    # 5) train 
    head.train()
    for _ in range(epochs):
        opt.zero_grad()

        scores = head(X_t.index_select(0, idx)).view(-1)  # (N,)
        y = rel_t.index_select(0, idx)                    # (N,)

        loss = pairwise_logistic_loss(scores, y)
        if loss.item() == 0.0:
            break

        loss.backward()
        opt.step()

    return head, feature_cols


def run_brand_pipeline(brand_id: str, train_df: pd.DataFrame,  test_df: pd.DataFrame,
    metric_modes: List[str], all_embeddings: Dict[str, np.ndarray] = None,
    all_metadata: pd.DataFrame = None, brand_embeddings: Dict[str, np.ndarray] = None, 
    use_cross_brand_for_features: bool = False, use_critique: bool = True, model_type: str = "lgbm"):

    train_df2 = train_df.copy()
    test_df2 = test_df.copy()
    for mode in metric_modes:
        if mode not in METRIC_MODES:
            raise ValueError(f"Unsupported metric mode: {mode}")
        target_col = METRIC_MODES[mode]["col"]
        train_df2 = add_target_metric(train_df2, clicks_col=CLICKS_COL, spend_col=SPEND_COL, imp_col=IMP_COL, mode=mode, out_col=target_col)
        test_df2 = add_target_metric(test_df2, clicks_col=CLICKS_COL, spend_col=SPEND_COL, imp_col=IMP_COL, mode=mode, out_col=target_col)
    cross_brand_reasoning = ""
    
    if use_cross_brand_for_features and all_metadata is not None and (all_embeddings or brand_embeddings):
        print(f"[{brand_id}] Stage 1.5: Looking for similar brand for cross-brand reasoning...")
        similar_brand_id, reasoning_samples = find_most_similar_brand(
            brand_id,
            all_metadata,
            metric_modes,
            all_embeddings=all_embeddings,
            brand_embeddings=brand_embeddings,
        )
        
        if similar_brand_id and reasoning_samples:
            print(f"[{brand_id}] Found similar brand {similar_brand_id} with {len(reasoning_samples)} samples")
            cross_brand_reasoning = build_cross_brand_reasoning(
                similar_brand_id,
                reasoning_samples,
                metric_modes,
            )
            if cross_brand_reasoning:
                print(f"[{brand_id}] ✓ Cross-brand reasoning generated and will be used for feature selection + LLM scoring")
                print(f"[{brand_id}] Reasoning preview: {cross_brand_reasoning}...")
        else:
            print(f"[{brand_id}] No similar brand found or no samples available")
    else:
        if not use_cross_brand_for_features:
            print(f"[{brand_id}] Skipping cross-brand reasoning (--use_cross_brand_for_features not set)")
        else:
            print(f"[{brand_id}] Skipping cross-brand reasoning (missing metadata or embeddings)")

    global BRAND_DESCRIPTIONS
    brand_description = BRAND_DESCRIPTIONS.get(brand_id, None)
    
    brand_performance_samples = extract_brand_performance_samples(train_df2,  metric_modes=metric_modes, num_per_tier=2)
    
    samples_info = f"with {len(brand_performance_samples)} brand samples" if brand_performance_samples else "without brand samples"
    insights_info = "with cross-brand insights" if cross_brand_reasoning else "without cross-brand insights"
    
    fewshot_examples_primary = build_fewshot_examples_multimetric(
        train_df2,
        metric_modes=metric_modes,
        max_examples=FEWSHOT_MAX_EXAMPLES,
        example_size=FEWSHOT_EXAMPLE_SIZE,
        target_metric=metric_modes[0], 
    )
    print(f"[{brand_id}] Built {len(fewshot_examples_primary)} few-shot examples for primary metric: {metric_modes[0]}")
    print(f"[{brand_id}] Stage 2: Selecting features ({insights_info}, {samples_info}, with few-shot examples)...")
    
    selected_features = select_features_for_brand(
        brand_id,
        num_features=4,
        brand_description=brand_description,
        cross_brand_reasoning=cross_brand_reasoning,
        brand_performance_samples=brand_performance_samples,
        metric_modes=metric_modes,
        fewshot_examples=fewshot_examples_primary,
    )
    print(f'Brand {brand_id} cross_brand_reasoning: {cross_brand_reasoning}')
    global BRAND_SELECTED_FEATURES
    BRAND_SELECTED_FEATURES[brand_id] = selected_features
    
    llm_feature_cols = [f"llm_{k}" for k in selected_features.keys()]

    print(f"[{brand_id}] Stage 3: Augmenting with GPT using COMMON few-shot examples")
    train_aug_common = augment_with_llm_features_multimodel(
        brand_id,
        train_df2,
        fewshot_examples=fewshot_examples_primary,
        metric_modes=metric_modes,
        selected_features=selected_features,
        cross_brand_reasoning=cross_brand_reasoning,
        use_critique=use_critique,
    )
    test_aug_common = augment_with_llm_features_multimodel(
        brand_id,
        test_df2,
        fewshot_examples=fewshot_examples_primary,
        metric_modes=metric_modes,
        selected_features=selected_features,
        cross_brand_reasoning=cross_brand_reasoning,
        use_critique=use_critique,
    )

    similar_ads_map = {}
    all_preds = []
    all_metrics = []

    for mode in metric_modes:
        target_col = METRIC_MODES[mode]["col"]
        direction = METRIC_MODES[mode]["direction"]
        need_cols = ["ad_id", "caption", target_col] + TABULAR_COLS_NUM + TABULAR_COLS_CAT + llm_feature_cols
        tr = train_aug_common[[c for c in need_cols if c in train_aug_common.columns]].dropna(subset=["ad_id", target_col]).copy()
        print(tr.columns)
        te = test_aug_common[[c for c in need_cols if c in test_aug_common.columns]].dropna(subset=["ad_id", target_col]).copy()

        if tr.empty or te.empty:
            print(f"[{brand_id}][{mode}] WARNING: train/test empty after filtering. Skipping.")
            continue

        all_model_types = ['mlp', 'linear', 'lgbm', 'llm']
        for model_type in all_model_types:
            if model_type == 'lgbm':
                print(f"[{brand_id}][{mode}] Training LGBM...")
                model, feat_cols = train_lgbm_ranker(tr, target_col, direction=direction, llm_feature_cols=llm_feature_cols, tune_hyperparams=False)
                pred_df = predict_lgbm_scores(model, feat_cols, te, llm_feature_cols=llm_feature_cols)
            elif model_type == 'mlp':
                print(f"[{brand_id}][{mode}] Training MLP...")
                head, feat_cols = train_pairwise_ranker_head(
                    tr, target_col=target_col, head_type="mlp", direction=direction, llm_feature_cols=llm_feature_cols
                )
                pred_df = predict_torch_ranker(head, te, feat_cols, llm_feature_cols=llm_feature_cols)
            elif model_type == 'linear':
                print(f"[{brand_id}][{mode}] Training Linear...")
                head, feat_cols = train_pairwise_ranker_head(
                    tr, target_col=target_col, head_type="linear", direction=direction, llm_feature_cols=llm_feature_cols
                )
                pred_df = predict_torch_ranker(head, te, feat_cols, llm_feature_cols=llm_feature_cols)
            elif model_type == 'llm':
                print(f"[{brand_id}][{mode}] Performing LLM Direct Ranking...")
                pred_df = llm_direct_ranking(brand_id=brand_id, df=te, target_col=target_col, direction=direction, selected_features=selected_features,
                    llm_feature_cols=llm_feature_cols, cross_brand_reasoning=cross_brand_reasoning, batch_size=10)
            else:
                print(f"[{brand_id}][{mode}] WARNING: Unknown model_type={model_type}. Skipping.")
                continue

            truth = te[["ad_id", target_col]].copy()
            truth["ad_id"] = truth["ad_id"].astype(str)
            metrics = eval_ndcg_kendall(pred_df.copy(), truth, target_col, direction=direction)

            feature_names = ", ".join(selected_features.keys())
            rag_suffix = " [RAG]" if similar_ads_map else ""
            print(
                f"[{brand_id}][{mode}][{model_type.upper()}] (dynamic LLM features={feature_names}, target={target_col}){rag_suffix}  "
                f"N1={metrics['ndcg@1']:.4f} | "
                f"N3={metrics['ndcg@3']:.4f} | "
                f"N5={metrics['ndcg@5']:.4f}"
            )

            stage_name = f"{model_type}_gpt_dynamic_llm_{target_col}"
            if similar_ads_map:
                stage_name += "_rag"
            
            pred_df_model = pred_df.assign(
                stage=stage_name,
                brand_id=brand_id,
                dw_account_id=brand_id,
                metric_mode=mode,
                target_col=target_col,
                selected_features=feature_names,
                model_type=model_type,
            )
            all_preds.append(pred_df_model)

            all_metrics.append(
                {
                    "dw_account_id": brand_id,
                    "stage": stage_name,
                    "metric_mode": mode,
                    "target_col": target_col,
                    "model_type": model_type,
                    **metrics,
                }
            )

    return all_preds, all_metrics

def main():
    global CSV_DIR, CAPTION_MAP, SEED, BRAND_DESCRIPTIONS

    parser = argparse.ArgumentParser(
        description="ADvisor"
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="data/",
        help="Directory containing train_*.csv and test_*.csv",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="ctr,spc,spi",
        help="Comma-separated target metrics: subset of {spc, ctr, spi}. Example: 'spc,ctr'",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="pipeline",
        help="Prefix for output CSV filenames",
    )
    parser.add_argument(
        "--caption_json",
        type=str,
        default="caption_dict.json",
        help="Path to JSON file mapping ad_id to caption (e.g., {'123': '...'}). "
             "If omitted, will try CSV_DIR/captions.json if it exists.",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="caption_embeddings.pkl",
        help="Path to pickle file containing pre-computed caption embeddings Dict[ad_id, embedding]",
    )
    parser.add_argument(
        "--brand_embedding_path",
        type=str,
        default="brand_embeddings.pkl",
        help="Path to pickle file containing pre-computed brand embeddings Dict[brand_id, embedding]. "
             "If provided, this will be used instead of averaging ad embeddings for brand similarity.",
    )
    parser.add_argument(
        "--brand_desc_json",
        type=str,
        default="brand_descriptions.json",
        help="Path to JSON file mapping brand_id to brand description (e.g., {'103931000000000': 'Luxury fashion brand...'})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use_cross_brand_for_features",
        action="store_true",
        default=False,
        help="If set, use cross-brand reasoning for feature selection. "
             "Analyzes similar brands' performance patterns to choose relevant features.",
    )
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="critique",
        choices=["single", "critique"],
        help="Scoring mode: 'single' for direct single-pass LLM scoring, "
             "'critique' for 3-stage (initial -> critique -> final) scoring. Default: critique",
    )
    args = parser.parse_args()
    # Apply seed
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Random seed set to: {SEED}")

    CSV_DIR = args.csv_dir
    metric_modes = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_modes:
        raise ValueError("No metrics specified. Use e.g. --metrics spc,ctr,spi")

    for m in metric_modes:
        if m not in METRIC_MODES:
            raise ValueError(f"Unsupported metric: {m}. Supported: {list(METRIC_MODES.keys())}")
    
    metric_info = "; ".join(f"{m}({'lower' if METRIC_MODES[m]['direction']=='lower' else 'higher'} is better)" for m in metric_modes)

    caption_json_path = args.caption_json
    if caption_json_path is None:
        default_path = os.path.join(CSV_DIR, "captions.json")
        if os.path.exists(default_path):
            caption_json_path = default_path

    CAPTION_MAP = load_caption_map(caption_json_path) if caption_json_path else {}

    if args.brand_desc_json and os.path.exists(args.brand_desc_json):
        print(f"Loading brand descriptions from {args.brand_desc_json}...")
        try:
            with open(args.brand_desc_json, "r", encoding="utf-8") as f:
                brand_desc_data = json.load(f)
            if isinstance(brand_desc_data, dict):
                BRAND_DESCRIPTIONS = {str(k): str(v) for k, v in brand_desc_data.items()}
                print(f"Loaded descriptions for {len(BRAND_DESCRIPTIONS)} brands")
            else:
                print("Warning: Brand description JSON is not a dict; ignoring.")
        except Exception as e:
            print(f"Warning: Failed to load brand descriptions: {e}")
    else:
        print("No brand description file provided (--brand_desc_json). Feature selection will use brand_id only.")

    all_embeddings = {}
    if args.embedding_path and os.path.exists(args.embedding_path):
        print(f"Loading embeddings from {args.embedding_path}...")
        all_embeddings = load_embeddings(args.embedding_path)
        print(f"Loaded {len(all_embeddings)} embeddings")
    else:
        print(f"Warning: Embedding file not found at {args.embedding_path} - RAG will be disabled")

    brand_embeddings = {}
    if args.brand_embedding_path and os.path.exists(args.brand_embedding_path):
        print(f"Loading brand embeddings from {args.brand_embedding_path}...")
        brand_embeddings = load_embeddings(args.brand_embedding_path)
        print(f"Loaded {len(brand_embeddings)} brand embeddings")
    else:
        if args.brand_embedding_path:
            print(f"Warning: Brand embedding file not found at {args.brand_embedding_path}")
        print("Will compute brand embeddings from ad embeddings if needed")

    print("=" * 80)
    print("4-STAGE PIPELINE with RAG: Meta Feature Discovery → Brand Feature Selection → LGBM")
    print("=" * 80)
    print("3-STAGE PIPELINE: Brand Feature Selection → LLM Scoring → Model Ranking")
    print("=" * 80)
    print(f"CSV_DIR      = {CSV_DIR}")
    print(f"METRICS      = {metric_modes}")
    print("TARGET_COLS  = " + ", ".join(f"{m}->{METRIC_MODES[m]['col']} ({METRIC_MODES[m]['direction']})" for m in metric_modes))
    print(f"DIRECTIONS   = {metric_info}")
    print("Augmentor    = GPT (3-stage with dynamic feature selection + LLM Direct Ranking)")
    print(f"Captions     = {len(CAPTION_MAP)} loaded from JSON" if CAPTION_MAP else "Captions     = using CSV-only captions (no external JSON)")
    print(f"Cross-Brand (feature selection) = {args.use_cross_brand_for_features} (--use_cross_brand_for_features flag)")
    print(f"Embeddings   = {len(all_embeddings)} loaded (RAG enabled)" if all_embeddings else "Embeddings   = None (RAG disabled)")
    print(f"Brand Embeds = {len(brand_embeddings)} loaded (pre-computed)" if brand_embeddings else "Brand Embeds = None (will average from ads if needed)")
    print(f"Brand Descs  = {len(BRAND_DESCRIPTIONS)} loaded" if BRAND_DESCRIPTIONS else "Brand Descs  = None (using brand_id only)")
    print(f"Few-shot     = max_examples={FEWSHOT_MAX_EXAMPLES}, example_size={FEWSHOT_EXAMPLE_SIZE}")
    print(f"Scoring mode = {args.scoring_mode} (3-stage critique={args.scoring_mode == 'critique'})")
    print(f"Seed         = {SEED}")
    print("=" * 80)

    train_files = [f for f in os.listdir(CSV_DIR) if re.match(r"train_\d+\.csv", f)]
    test_files  = [f for f in os.listdir(CSV_DIR) if re.match(r"test_\d+\.csv", f)]
    brands = sorted(set(re.findall(r"\d+", " ".join(train_files + test_files))))

    brand_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for bid in brands:
        tfn = os.path.join(CSV_DIR, f"train_{bid}.csv")
        vfn = os.path.join(CSV_DIR, f"test_{bid}.csv")
        if os.path.exists(tfn) and os.path.exists(vfn):
            tr = pd.read_csv(tfn)
            te = pd.read_csv(vfn)
            brand_splits[bid] = {"train": tr, "test": te}

    print(f"\nLoaded {len(brand_splits)} brands")

    all_metadata = None
    if all_embeddings:
        print("\nPreparing metadata for RAG (combining all brand data)...")
        metadata_list = []
        for bid, splits in brand_splits.items():
            tr = splits["train"].copy()
            te = splits["test"].copy()
            tr["brand_id"] = bid
            te["brand_id"] = bid
            metadata_list.append(tr)
            metadata_list.append(te)
        
        if metadata_list:
            all_metadata = pd.concat(metadata_list, ignore_index=True)
            print(f"Combined metadata: {len(all_metadata)} rows from {len(brand_splits)} brands")

    all_preds, all_metrics = [], []
    use_critique = args.scoring_mode == "critique"
    print(f"Scoring mode = {args.scoring_mode} (use_critique={use_critique})")
    
    for bid, splits in brand_splits.items():
        print(f"\n{'='*80}")
        print(f"BRAND {bid}")
        print(f"{'='*80}")
        preds_list, metrics_list = run_brand_pipeline(
            bid, 
            splits["train"], 
            splits["test"], 
            metric_modes, 
            all_embeddings=all_embeddings if all_embeddings else None,
            all_metadata=all_metadata,
            brand_embeddings=brand_embeddings if brand_embeddings else None,
            use_cross_brand_for_features=args.use_cross_brand_for_features,
            use_critique=use_critique,
        )
        all_preds.extend(preds_list)
        all_metrics.extend(metrics_list)

        preds_out = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
        metrics_out = pd.DataFrame(all_metrics)
        if not metrics_out.empty:
            metrics_out = metrics_out.sort_values(["dw_account_id", "metric_mode", "stage", "model_type"])

    preds_out_path = f"{args.output_prefix}_ADVISOR.csv"
    metrics_out_path = f"{args.output_prefix}_metrics_ADVISOR.csv"
    
    brand_features_path = f"{args.output_prefix}_brand_features.json"
    try:
        with open(brand_features_path, "w", encoding="utf-8") as f:
            json.dump(BRAND_SELECTED_FEATURES, f, indent=2, ensure_ascii=False)
        print(f"\nBrand-specific features saved to: {brand_features_path}")
    except Exception as e:
        print(f"\nWarning: Could not save brand features: {e}")

    preds_out.to_csv(preds_out_path, index=False)
    metrics_out.to_csv(metrics_out_path, index=False)

    print(f"\nFinal outputs saved:")
    print(f"  - {preds_out_path}")
    print(f"  - {metrics_out_path}")

    print("\n" + "="*80)
    print("GENERATING PERFORMANCE SUMMARY TABLE")
    print("="*80)
    
    metric_cols = ["ndcg@1", "ndcg@3", "ndcg@5"]
    metric_mode_mapping = {"ctr": "CTR", "spc": "CPC", "spi": "CPM"}
    model_types_for_summary = sorted(metrics_out["model_type"].dropna().unique())
    
    for model_type in model_types_for_summary:
        if model_type is None:
            metrics_slice = metrics_out
            model_suffix = ""
        else:
            metrics_slice = metrics_out[metrics_out["model_type"] == model_type]
            model_suffix = f"_{model_type}"
        
        if metrics_slice.empty:
            continue

        print("\n" + "-" * 80)
        print(f"PERFORMANCE SUMMARY TABLE for MODEL: {model_type}")
        print("-" * 80)

        summary_data = []
        for brand_id in sorted(metrics_slice["dw_account_id"].unique()):
            brand_data = metrics_slice[metrics_slice["dw_account_id"] == brand_id]
            
            for metric_col_name in metric_cols:
                row_dict = {"BrandId": brand_id, "Metric": metric_col_name}
                
                for metric_mode in metric_modes:
                    mode_data = brand_data[brand_data["metric_mode"] == metric_mode]
                    if not mode_data.empty:
                        value = mode_data[metric_col_name].iloc[0]
                        col_name = metric_mode_mapping.get(metric_mode, metric_mode.upper())
                        row_dict[col_name] = float(value) if pd.notna(value) else 0.0
                    else:
                        col_name = metric_mode_mapping.get(metric_mode, metric_mode.upper())
                        row_dict[col_name] = 0.0
                
                summary_data.append(row_dict)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Reorder columns
            metric_cols_display = ["BrandId", "Metric"] + [metric_mode_mapping.get(m, m.upper()) for m in metric_modes]
            summary_df = summary_df[metric_cols_display]
            
            # Calculate row average (average across metric modes for each brand/metric combo)
            metric_value_cols = [metric_mode_mapping.get(m, m.upper()) for m in metric_modes]
            summary_df["Row_Avg"] = summary_df[metric_value_cols].mean(axis=1)
            
            # Add brand-specific average rows
            all_rows_with_brand_avg = [summary_df.copy()]
            
            for brand_id in sorted(metrics_slice["dw_account_id"].unique()):
                brand_summary = summary_df[summary_df["BrandId"] == brand_id]
                if not brand_summary.empty:
                    brand_avg_row = {"BrandId": brand_id, "Metric": "Brand_Avg"}
                    for col in metric_value_cols:
                        brand_avg_row[col] = brand_summary[col].mean()
                    brand_avg_row["Row_Avg"] = brand_summary[metric_value_cols].values.flatten().mean()
                    all_rows_with_brand_avg.append(pd.DataFrame([brand_avg_row]))
            
            # Combine all rows
            summary_df = pd.concat(all_rows_with_brand_avg, ignore_index=True)
            
            # Calculate column averages (average across all brands for each metric)
            data_only_df = summary_df[~summary_df["Metric"].str.contains("Brand_Avg", na=False)]
            col_avg_row = {"BrandId": "AVG_PER_METRIC", "Metric": ""}
            for col in metric_value_cols:
                col_avg_row[col] = data_only_df[col].mean()
            col_avg_row["Row_Avg"] = data_only_df[metric_value_cols].values.flatten().mean()
            
            # Calculate overall average
            overall_avg_row = {"BrandId": "OVERALL_AVG", "Metric": ""}
            for col in metric_value_cols:
                overall_avg_row[col] = data_only_df[col].mean()
            overall_avg_row["Row_Avg"] = data_only_df[metric_value_cols].values.flatten().mean()
            
            # Append average rows
            summary_df = pd.concat([
                summary_df,
                pd.DataFrame([col_avg_row]),
                pd.DataFrame([overall_avg_row])
            ], ignore_index=True)
            
        else:
            print("Warning: No summary data generated for this model_type.")


if __name__ == "__main__":
    main()
