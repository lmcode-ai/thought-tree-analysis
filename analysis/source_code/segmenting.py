import json
import spacy
from typing import List, Dict
from statistics import mean
from pathlib import Path
import argparse
import os

nlp = spacy.load("en_core_web_sm")

DATASET_CONFIG = {
    "input_path": "../../llm_evaluation/{dataset}/{model}_{dataset}/{task}_{level}.json",
    "output_path": "../{dataset}/raw_segments1/{model}_{level}_{task}_segmented_reasoning.json",
}

WEAK_STARTERS = {
    "so", "but", "also", "then", "thus", "therefore", "meanwhile", "now",
    "okay", "however", "next", "first", "second", "wait", "although"
}

def is_code_like(text: str) -> bool:
    return any(sym in text for sym in ['=', '{', '}', ';', '->', '==', '!=', 'for (', 'while ('])

def segment(text: str, seg_len=100, min_len=30) -> List[Dict]:
    doc = nlp(text)
    segments = []
    buffer = ""
    buffer_start = None

    for i, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        first_word = sent_text.split()[0].lower() if sent_text.split() else ""
        is_weak_start = first_word in WEAK_STARTERS
        is_short = len(sent_text) < min_len
        is_code = is_code_like(sent_text)

        prev_ends_code = buffer.strip().endswith((";", "}")) if buffer else False

        if not buffer:
            buffer = sent_text
            buffer_start = sent.start_char
            continue

        # Heuristic to merge if short or weak, or current is small and last ended with code
        if (is_short or is_weak_start or prev_ends_code) and not is_code:
            buffer += " " + sent_text
        else:
            clean = buffer.strip()
            segments.append({
                "text": clean,
                "start_char": buffer_start,
                "end_char": buffer_start + len(clean)
            })
            buffer = sent_text
            buffer_start = sent.start_char

    if buffer:
        clean = buffer.strip()
        segments.append({
            "text": clean,
            "start_char": buffer_start,
            "end_char": buffer_start + len(clean)
        })

    # Post-processing: merge segments that are too short or dangling code
    merged_segments = []
    for seg in segments:
        if not merged_segments:
            merged_segments.append(seg)
            continue
        prev = merged_segments[-1]
        if len(seg["text"]) < seg_len or is_code_like(seg["text"]):
            # Merge into previous
            prev["text"] += " " + seg["text"]
            prev["end_char"] = seg["end_char"]
        else:
            merged_segments.append(seg)

    return merged_segments

def run_segmentation(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = json.load(f)

    result = {}
    all_segment_counts = []
    all_segment_lengths = []

    for qid, entry in data.items():
        reasoning = entry.get("reasoning", "")
        # reasoning = entry
        segments = segment(reasoning)

        all_segment_counts.append(len(segments))
        all_segment_lengths.extend([len(seg["text"]) for seg in segments])

        result[qid] = {
            "segments": segments,
            "original_reasoning": reasoning,
            "metadata": {
                # "state": entry.get("state", ""),
                "score": entry.get("score", -1),
                # "code": entry.get("code", ""),
                "answer": entry.get("answer", "")
            }
        }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n Saved segmented reasoning traces to: {output_path}")
    print("\n--- Segmentation Stats ---")
    print(f"Entries: {len(result)}")
    print(f"Avg segments per entry: {mean(all_segment_counts):.2f}")
    print(f"Avg segment length (chars): {mean(all_segment_lengths):.2f}")
    print(f"Total segments: {len(all_segment_lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation on a specific JSON file.")

    parser.add_argument("--dataset", type=str, required=True, help="Key from DATASET_CONFIG (e.g., 'crux_eval')")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'output', 'detection')")
    parser.add_argument("--level", type=str, required=True, help="Level (e.g., 'level1', 'hard')")
    parser.add_argument("--model", type=str, required=True, help="Short model name (e.g., 'qwq', 'r1')")
    args = parser.parse_args()

    # Run the function
    input_path = DATASET_CONFIG["input_path"].format(dataset=args.dataset, model=args.model, level=args.level, task=args.task)
    output_path = DATASET_CONFIG["output_path"].format(dataset=args.dataset, model=args.model, level=args.level, task=args.task)
    print(f"Processing: {input_path} -> {output_path}")
    run_segmentation(input_path, output_path)