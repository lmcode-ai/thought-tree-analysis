import json
from statistics import mean
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
import re
import numpy as np
import argparse

# Define your model constants if needed
DEEPSEEK_R1 = "r1"
QWQ = "qwq"

# Define different path structures for your datasets here
DATASET_CONFIG = {
    "trace_json": "../../llm_evaluation/{dataset}/{model}_{dataset}/{task}_{level}.json",
    "label_dir": "../{dataset}/segment_labeled/{model}_{level}_{task}",
    "tree_dir": "../{dataset}/segment_trees/{model}_{level}_{task}",
    "output_csv": "../{dataset}/tree_stats/{model}_{level}_{task}.csv",
    "plot_hist": "../{dataset}/plots/{model}_{level}_{task}_histogram.png",
    "plot_violin": "../{dataset}/plots/{model}_{level}_{task}_violin.png"
}

def process_labeled_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        return {"_error": f"failed_load: {e}"}

    segments = data.get('segments', [])

    # Convert this file's segments to a DataFrame for easy math
    df_seg = pd.DataFrame(segments)
    
    # 1. Determine Lengths (use char indices if available, else text length)
    if 'start_char' in df_seg.columns and 'end_char' in df_seg.columns:
        lengths = df_seg['end_char'] - df_seg['start_char']
    elif 'text' in df_seg.columns:
        lengths = df_seg['text'].str.len()
    else:
        lengths = pd.Series([0] * len(df_seg))

    # 2. Build the statistics dictionary for this file
    labels = ["code_analysis", "mental_execution", "test_generation", "bug_fixing", "language_mapping", "high_level_plan", "empty"]
    label_counts = {label: 0 for label in labels}
    actual_counts = df_seg['label'].value_counts().to_dict() if 'label' in df_seg.columns else {}
    if actual_counts:
        for l, c in actual_counts.items():
            label_counts[l] = c

    stats = {
        'filename': Path(path).name,
        'total_segments': len(df_seg),
        # 'unique_labels_count': df_seg['label'].nunique() if 'label' in df_seg.columns else 0,
        
        # Length Metrics
        'avg_segment_len': float(lengths.mean()),
        'max_segment_len': float(lengths.max()),
        'min_segment_len': float(lengths.min()),
        'std_dev_len': float(lengths.std()),  # Standard deviation shows how much lengths vary
    }
    for label_name, count in label_counts.items():
        stats[label_name] = int(count)
    return stats

def extract_features_v3(reasoning: str, dataset_name: str = ""):
    """V3: Enhanced features with task metadata."""
    features = {}

    if not reasoning or not reasoning.strip():
        # Return all zeros
        return {
            'log_length': 0, 'log_word_count': 0, 'avg_sent_len': 0, 'sent_len_cv': 0,
            'hesitation_rate': 0, 'correction_rate': 0, 'confidence_rate': 0,
            'conf_hesit_ratio': 0, 'backtrack_rate': 0, 'forward_rate': 0,
            'fw_bt_ratio': 0, 'question_rate': 0, 'code_density': 0,
            'early_hesit': 0, 'mid_hesit': 0, 'late_hesit': 0, 'hesit_trend': 0,
            'early_correct': 0, 'late_correct': 0, 'correct_trend': 0,
            'analysis_rate': 0, 'exec_rate': 0, 'plan_rate': 0, 'verify_rate': 0,
            'strategy_entropy': 0, 'lex_diversity': 0, 'conclusion_strength': 0,
            'first_person_rate': 0, 'paragraph_rate': 0,
            'success_words': 0, 'failure_words': 0, 'success_fail_ratio': 0,
            # 'is_fortran': 0, 'is_level_hard': 0,
        }
    
    text = reasoning.strip()
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = max(len(sentences), 1)
    
    # === BASIC (log-normalized) ===
    features['log_length'] = np.log1p(len(text))
    features['log_word_count'] = np.log1p(word_count)

    
    sent_lens = [len(s.split()) for s in sentences] if sentences else [0]
    features['avg_sent_len'] = np.mean(sent_lens)
    features['sent_len_cv'] = np.std(sent_lens) / max(np.mean(sent_lens), 1)
    
    # === HESITATION & UNCERTAINTY ===
    hesit_words = ["wait", "hmm", "maybe", "perhaps", "actually", "hold on", 
                   "confused", "unclear", "not sure", "i think"]
    hesit_count = sum(text_lower.count(w) for w in hesit_words)
    features['hesitation_rate'] = hesit_count / word_count * 1000 if word_count else 0
    
    # === SELF-CORRECTION ===
    correct_patterns = [
        r"no[,.]?\s*(that'?s)?\s*(wrong|incorrect)",
        r"\bactually[,.]",
        r"\bwait[,.]?\s*(no|that)",
        r"let me (re|start|try)",
        r"my mistake",
        r"\boops\b",
    ]
    correct_count = sum(len(re.findall(p, text_lower)) for p in correct_patterns)
    features['correction_rate'] = correct_count / word_count * 1000 if word_count else 0
    
    # === CONFIDENCE ===
    conf_words = ["clearly", "obviously", "definitely", "certainly", "straightforward",
                  "simple", "easy", "done", "works", "correct"]
    conf_count = sum(text_lower.count(w) for w in conf_words)
    features['confidence_rate'] = conf_count / word_count * 1000 if word_count else 0
    features['conf_hesit_ratio'] = (conf_count + 1) / (hesit_count + 1)
    
    # === BACKTRACK vs FORWARD ===
    bt_words = ["wait", "no,", "actually", "hmm", "oops"]
    fw_words = ["so,", "therefore", "thus", "hence", "next,", "then,", "finally"]
    
    bt_count = sum(text_lower.count(w) for w in bt_words)
    fw_count = sum(text_lower.count(w) for w in fw_words)
    
    features['backtrack_rate'] = bt_count / word_count * 1000 if word_count else 0
    features['forward_rate'] = fw_count / word_count * 1000 if word_count else 0
    features['fw_bt_ratio'] = (fw_count + 1) / (bt_count + 1)
    
    features['question_rate'] = text.count('?') / sent_count
    
    # === CODE DENSITY ===
    code_pats = [r'\b(int|float|char|void)\s+\w+', r'\b(for|while|if)\s*\(', r'[{};]']
    code_matches = sum(len(re.findall(p, text)) for p in code_pats)
    features['code_density'] = code_matches / word_count * 100 if word_count else 0
    
    # === 3-PHASE ANALYSIS (early/mid/late) ===
    third = len(text) // 3
    early = text_lower[:third]
    mid = text_lower[third:2*third]
    late = text_lower[2*third:]
    
    early_words = len(early.split()) or 1
    mid_words = len(mid.split()) or 1
    late_words = len(late.split()) or 1
    
    features['early_hesit'] = sum(early.count(w) for w in hesit_words) / early_words * 1000
    features['mid_hesit'] = sum(mid.count(w) for w in hesit_words) / mid_words * 1000
    features['late_hesit'] = sum(late.count(w) for w in hesit_words) / late_words * 1000
    features['hesit_trend'] = features['late_hesit'] - features['early_hesit']
    
    # Corrections by phase
    def count_corrections(txt):
        return sum(len(re.findall(p, txt)) for p in correct_patterns)
    
    features['early_correct'] = count_corrections(early) / early_words * 1000
    features['late_correct'] = count_corrections(late) / late_words * 1000
    features['correct_trend'] = features['late_correct'] - features['early_correct']
    
    # === STRATEGY DETECTION ===
    analysis_pats = [r'(this|the) (code|function|loop)', r'looking at', r'analyzing']
    exec_pats = [r"if\s+(we|n|x)\s*=", r"for example", r"simulate"]
    plan_pats = [r"^(first|step \d)", r"i('ll| will| need to)"]
    verify_pats = [r"let me (check|verify|test)", r"make sure"]
    
    features['analysis_rate'] = sum(len(re.findall(p, text_lower)) for p in analysis_pats) / sent_count
    features['exec_rate'] = sum(len(re.findall(p, text_lower)) for p in exec_pats) / sent_count
    features['plan_rate'] = sum(len(re.findall(p, text_lower, re.MULTILINE)) for p in plan_pats) / sent_count
    features['verify_rate'] = sum(len(re.findall(p, text_lower)) for p in verify_pats) / sent_count
    
    # Strategy entropy (diversity)
    strat_counts = [features['analysis_rate'], features['exec_rate'], 
                    features['plan_rate'], features['verify_rate']]
    total = sum(strat_counts) + 0.001
    probs = [c/total for c in strat_counts]
    features['strategy_entropy'] = -sum(p * np.log(p + 0.001) for p in probs)
    
    # === LEXICAL ===
    unique = set(w.lower() for w in words if w.isalpha() and len(w) > 2)
    features['lex_diversity'] = len(unique) / word_count if word_count else 0
    
    # === CONCLUSION ===
    concl_words = ["finally", "done", "complete", "that's it", "solution"]
    features['conclusion_strength'] = sum(text_lower.count(w) for w in concl_words)
    
    # === FIRST PERSON ===
    fp_words = ["i ", "i'm", "i'll", "my "]
    features['first_person_rate'] = sum(text_lower.count(w) for w in fp_words) / word_count * 100 if word_count else 0
    
    # === STRUCTURE ===
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    features['paragraph_rate'] = paragraphs / (word_count / 100) if word_count else 0
    
    # === SUCCESS/FAILURE LANGUAGE ===
    success_words = ["works", "correct", "right", "good", "done", "solved", "success"]
    failure_words = ["wrong", "error", "bug", "fail", "issue", "problem", "broken"]
    
    succ_count = sum(text_lower.count(w) for w in success_words)
    fail_count = sum(text_lower.count(w) for w in failure_words)
    
    features['success_words'] = succ_count / word_count * 1000 if word_count else 0
    features['failure_words'] = fail_count / word_count * 1000 if word_count else 0
    features['success_fail_ratio'] = (succ_count + 1) / (fail_count + 1)
    
    # # === TASK METADATA (generalizable) ===
    # features['is_fortran'] = 1.0 if 'Fortran' in dataset_name else 0.0
    # features['is_level_hard'] = 1.0 if any(x in dataset_name for x in ['L3', 'L5']) else 0.0
    
    return features

def tree_stats(root: dict) -> dict:
    stats = {
        "total_nodes": 0,
        "leaf_nodes": 0,
        "internal_nodes": 0,
        "max_depth": 0,
        "branching_factors": [],
    }

    def walk(node, depth=0):
        stats["total_nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)

        children = node.get("children", [])
        if children:
            stats["internal_nodes"] += 1
            stats["branching_factors"].append(len(children))
            for child in children:
                walk(child, depth + 1)
        else:
            stats["leaf_nodes"] += 1

    walk(root)

    stats["average_branching_factor"] = round(mean(stats["branching_factors"]), 2) if stats["branching_factors"] else 0
    stats["max_branching_factor"] = max(stats["branching_factors"], default=0)
    stats["min_branching_factor"] = min(stats["branching_factors"], default=0)
    stats["first_branching_factor"] = stats["branching_factors"][0] if stats["branching_factors"] else 0
    stats["depth_per_nodes"] = stats["max_depth"] / stats["total_nodes"] if stats["total_nodes"] else 0
    stats["internal_per_depth"] = stats["internal_nodes"] / (stats["max_depth"] if stats["max_depth"] else 1)
    return stats

def gen_plot_violin(data, label_column="state", save_path=None):
    data["label"] = data[label_column]
    metrics = [
        "max_depth",
        "total_nodes",
        "average_branching_factor",
        "first_branching_factor",
        "depth_per_nodes",
        "internal_per_depth",
        "internal_nodes",
    ]
    unique_labels = sorted(data["label"].unique())
    fig2, axes2 = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3))
    for ax, metric in zip(axes2, metrics):
        data_1 = [data.loc[data["label"] == label, metric] for label in unique_labels]
        positions = list(range(1, len(unique_labels) + 1))
        ax.violinplot(data_1, positions=positions, showmeans=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(unique_labels, rotation=15)
        ax.set_title(metric)
        ax.set_ylabel(metric)
    fig2.suptitle("Violin Plots")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def gen_plot_histogram(data, label_column="state", save_path=None):
    data["label"] = data[label_column]
    metrics = [
        "max_depth",
        "total_nodes",
        "average_branching_factor",
        "first_branching_factor",
        "depth_per_nodes",
        "internal_per_depth",
        "internal_nodes",
    ]
    unique_labels = sorted(data["label"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3), sharey=False)
    for ax, metric in zip(axes, metrics):
        for lbl in unique_labels:
            ax.hist(
                data.loc[data["label"] == lbl, metric],
                bins=20,
                alpha=0.6,
                label=lbl,
                density=True,
            )
        ax.set_title(metric)
        ax.set_xlabel("value")
        ax.legend()
        ax.autoscale(enable=True, axis="x", tight=True)
    fig.suptitle("Histograms")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)


def evaluate_tree(dataset_name, task, level, model, max_samples=None):
    file_scored = DATASET_CONFIG["trace_json"].format(dataset=dataset_name, model=model, task=task, level=level)
    label_dir = DATASET_CONFIG["label_dir"].format(model=model, level=level, task=task, dataset=dataset_name)
    tree_dir = DATASET_CONFIG["tree_dir"].format(model=model, level=level, task=task, dataset=dataset_name)
    output_csv = DATASET_CONFIG["output_csv"].format(model=model, level=level, task=task, dataset=dataset_name)
    plot_violin_path = DATASET_CONFIG["plot_violin"].format(model=model, level=level, task=task, dataset=dataset_name)
    plot_hist_path = DATASET_CONFIG["plot_hist"].format(model=model, level=level, task=task, dataset=dataset_name)

    print(f"--- Running Evaluation ---")
    print(f"Dataset: {dataset_name}")
    print(f"Tree Dir: {tree_dir}")
    print(f"Scored File: {file_scored}")

    try:
        with open(file_scored, "r") as f:
            data_generation_scored = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Scored file not found at {file_scored}. Proceeding without scores.")
        raise

    files = glob.glob(os.path.join(tree_dir, "qid_*.json"))
    files_to_process = files if max_samples is None else files[:int(max_samples)]
    print(f"Found {len(files)} files, processing {len(files_to_process)}...")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    results = {}
    for file in files_to_process:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                
            qid = data.get("qid", "")
            tree = data.get("tree", {})
            reasoning = tree.get("original_reasoning", "")
            
            # Extract features
            feat_reasoning = extract_features_v3(reasoning, dataset_name)
            stats_tree = tree_stats(tree)
            
            score = data_generation_scored.get(qid).get("score")
            
            if score == 1:
                state = "correct"
            elif score == 0:
                state = "incorrect"
            else:
                state = "unknown"

            results[qid] = {"file": file,
                            "total_nodes": stats_tree["total_nodes"],
                            "leaf_nodes": stats_tree["leaf_nodes"],
                            "internal_nodes": stats_tree["internal_nodes"],
                            "max_depth": stats_tree["max_depth"],
                            "average_branching_factor": stats_tree["average_branching_factor"],
                            "max_branching_factor": stats_tree["max_branching_factor"],
                            "min_branching_factor": stats_tree["min_branching_factor"],
                            "first_branching_factor": stats_tree["first_branching_factor"],
                            "depth_per_nodes": stats_tree["depth_per_nodes"],
                            "internal_per_depth": stats_tree["internal_per_depth"],
                            "state": state,
                                }
        except Exception as e:
            print(f"Error processing {file}: {e}")

    labeled_files = list(Path(label_dir).glob("*.json"))
    labeled_files = labeled_files if max_samples is None else labeled_files[:int(max_samples)]
    
    print(f"Found {len(labeled_files)} labeled files. Adding label features...")
    
    for file in labeled_files:
        file_id = file.stem 
        qid = file_id.replace("qid_", "")
        if qid in results:
            feat_labeled = process_labeled_file(file)
            results[qid].update(feat_labeled)
        else:
            # Try matching by filename substring if exact key fails
            for r_key in results:
                if r_key in file.stem or file.stem in r_key:
                    feat_labeled = process_labeled_file(file)
                    results[r_key].update(feat_labeled)
                    break

    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results.values())
    
    # Cleanup columns
    df = df.drop(columns=["filename", "error", "awaiting_segment", "...", "explanation"], errors='ignore')
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved stats to {output_csv}")
    
    os.makedirs(os.path.dirname(plot_violin_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_hist_path), exist_ok=True)
    gen_plot_violin(df, label_column="state", save_path=plot_violin_path)
    gen_plot_histogram(df, label_column="state", save_path=plot_hist_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on specific dataset, model, and task.")
    
    parser.add_argument("--dataset", type=str, required=True, help="Key from DATASET_CONFIG (e.g., 'crux_eval')")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'output', 'detection')")
    parser.add_argument("--level", type=str, required=True, help="Level (e.g., 'level1', 'hard')")
    parser.add_argument("--model", type=str, required=True, help="Short model name (e.g., 'qwq', 'r1')")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of files to process for testing")

    args = parser.parse_args()

    evaluate_tree(
        dataset_name=args.dataset,
        task=args.task,
        level=args.level,
        model=args.model,
        max_samples=args.max_samples
    )