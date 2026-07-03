import argparse
import json
import os
import random
import re

import numpy as np
import pandas as pd

from advisor import utils


def main():
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
    utils.SEED = args.seed
    random.seed(utils.SEED)
    np.random.seed(utils.SEED)
    print(f"Random seed set to: {utils.SEED}")

    utils.CSV_DIR = args.csv_dir
    metric_modes = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_modes:
        raise ValueError("No metrics specified. Use e.g. --metrics spc,ctr,spi")

    for m in metric_modes:
        if m not in utils.METRIC_MODES:
            raise ValueError(f"Unsupported metric: {m}. Supported: {list(utils.METRIC_MODES.keys())}")
    
    metric_info = "; ".join(f"{m}({'lower' if utils.METRIC_MODES[m]['direction']=='lower' else 'higher'} is better)" for m in metric_modes)

    caption_json_path = args.caption_json
    if caption_json_path and not os.path.exists(caption_json_path):
        default_path = os.path.join(utils.CSV_DIR, "captions.json")
        if os.path.exists(default_path):
            caption_json_path = default_path
        else:
            print(f"Warning: Caption file not found at {caption_json_path}; using CSV-only captions")
            caption_json_path = None
    elif caption_json_path is None:
        default_path = os.path.join(utils.CSV_DIR, "captions.json")
        if os.path.exists(default_path):
            caption_json_path = default_path

    utils.CAPTION_MAP = utils.load_caption_map(caption_json_path) if caption_json_path else {}

    if args.brand_desc_json and os.path.exists(args.brand_desc_json):
        print(f"Loading brand descriptions from {args.brand_desc_json}...")
        try:
            with open(args.brand_desc_json, "r", encoding="utf-8") as f:
                brand_desc_data = json.load(f)
            if isinstance(brand_desc_data, dict):
                utils.BRAND_DESCRIPTIONS = {str(k): str(v) for k, v in brand_desc_data.items()}
                print(f"Loaded descriptions for {len(utils.BRAND_DESCRIPTIONS)} brands")
            else:
                print("Warning: Brand description JSON is not a dict; ignoring.")
        except Exception as e:
            print(f"Warning: Failed to load brand descriptions: {e}")
    else:
        print("No brand description file provided (--brand_desc_json). Feature selection will use brand_id only.")

    all_embeddings = {}
    if args.embedding_path and os.path.exists(args.embedding_path):
        print(f"Loading embeddings from {args.embedding_path}...")
        all_embeddings = utils.load_embeddings(args.embedding_path)
        print(f"Loaded {len(all_embeddings)} embeddings")
    else:
        print(f"Warning: Embedding file not found at {args.embedding_path} - RAG will be disabled")

    brand_embeddings = {}
    if args.brand_embedding_path and os.path.exists(args.brand_embedding_path):
        print(f"Loading brand embeddings from {args.brand_embedding_path}...")
        brand_embeddings = utils.load_embeddings(args.brand_embedding_path)
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
    print(f"CSV_DIR      = {utils.CSV_DIR}")
    print(f"METRICS      = {metric_modes}")
    print("TARGET_COLS  = " + ", ".join(f"{m}->{utils.METRIC_MODES[m]['col']} ({utils.METRIC_MODES[m]['direction']})" for m in metric_modes))
    print(f"DIRECTIONS   = {metric_info}")
    print("Augmentor    = GPT (3-stage with dynamic feature selection + LLM Direct Ranking)")
    print(f"Captions     = {len(utils.CAPTION_MAP)} loaded from JSON" if utils.CAPTION_MAP else "Captions     = using CSV-only captions (no external JSON)")
    print(f"Cross-Brand (feature selection) = {args.use_cross_brand_for_features} (--use_cross_brand_for_features flag)")
    print(f"Embeddings   = {len(all_embeddings)} loaded (RAG enabled)" if all_embeddings else "Embeddings   = None (RAG disabled)")
    print(f"Brand Embeds = {len(brand_embeddings)} loaded (pre-computed)" if brand_embeddings else "Brand Embeds = None (will average from ads if needed)")
    print(f"Brand Descs  = {len(utils.BRAND_DESCRIPTIONS)} loaded" if utils.BRAND_DESCRIPTIONS else "Brand Descs  = None (using brand_id only)")
    print(f"Few-shot     = max_examples={utils.FEWSHOT_MAX_EXAMPLES}, example_size={utils.FEWSHOT_EXAMPLE_SIZE}")
    print(f"Scoring mode = {args.scoring_mode} (3-stage critique={args.scoring_mode == 'critique'})")
    print(f"Seed         = {utils.SEED}")
    print("=" * 80)

    if not os.path.isdir(utils.CSV_DIR):
        print(f"Warning: CSV directory not found at {utils.CSV_DIR}; no brands will be processed")
        train_files = []
        test_files = []
    else:
        train_files = [f for f in os.listdir(utils.CSV_DIR) if re.match(r"train_\d+\.csv", f)]
        test_files  = [f for f in os.listdir(utils.CSV_DIR) if re.match(r"test_\d+\.csv", f)]
    brands = sorted(set(re.findall(r"\d+", " ".join(train_files + test_files))))

    brand_splits: Dict[str, Dict[str, pd.DataFrame]] = {}
    for bid in brands:
        tfn = os.path.join(utils.CSV_DIR, f"train_{bid}.csv")
        vfn = os.path.join(utils.CSV_DIR, f"test_{bid}.csv")
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
    preds_out = pd.DataFrame()
    metrics_out = pd.DataFrame()
    use_critique = args.scoring_mode == "critique"
    print(f"Scoring mode = {args.scoring_mode} (use_critique={use_critique})")
    
    for bid, splits in brand_splits.items():
        print(f"\n{'='*80}")
        print(f"BRAND {bid}")
        print(f"{'='*80}")
        preds_list, metrics_list = utils.run_brand_pipeline(
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
            json.dump(utils.BRAND_SELECTED_FEATURES, f, indent=2, ensure_ascii=False)
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
    
    if metrics_out.empty or "model_type" not in metrics_out.columns:
        print("No metrics generated; skipping performance summary table.")
        return

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
