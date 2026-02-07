import json
import os
import time
from collections import defaultdict
from vertexai import generative_models
import vertexai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

API = "api_completion"
BLOCK = "block_completion"
CONTROL = "control_completion"
DEEPSEEK_R1 = "deepseek_r1"
GEMINI_PRO = "gemini_pro"
QWQ = "qwq"

PROJECT_ID = "goog24-12"      
LOCATION = "us-central1"    
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = generative_models.GenerativeModel("gemini-1.5-pro")

DEEPSEEK_R1_TOKENIZER = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1") 
QWQ_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/QwQ-32B")

PROMPT_TEMPLATE = """
You are an expert in reasoning trace analysis.

A reasoning trace describes the thought process of a model while solving a code translation task. Each trace may include multiple types of reasoning behaviors.

Your task is to:
1. Identify every reasoning strategy used
2. Count how many times each strategy appears (number of distinct segments or shifts)
3. Estimate the proportion of the trace that each strategy occupies (sum should be ~1.0)

Choose from this strategy list:

- "code_analysis" — analyzing what the original code does
- "mental_execution" — simulating input/output behavior
- "test_generation" — proposing test cases or edge conditions
- "bug_fixing" — describing an error message or traceback and then correcting syntax or logic errors
- "language_mapping" — translating code constructs from one language to another
- "high_level_plan" — outlining a step-by-step approach or plan
- "empty" — no meaningful content

Return only a JSON object like this:

```json
{{
  "reasoning_strategies": [
    {{"type": "...", "occurrences": ..., "proportion": ...}},
    ...
  ],
  "notes": "..."
}}

Reasoning:
{reasoning}
"""

def count_tokens(text, model):
    if model == DEEPSEEK_R1:
        return len(DEEPSEEK_R1_TOKENIZER.encode(text, add_special_tokens=False))
    elif model == QWQ:
        return len(QWQ_TOKENIZER.encode(text, add_special_tokens=False))

def load_jsonl_if_exists(path):
    if os.path.exists(path):
        with open(path, "r") as json_file:
            return [json.loads(line) for line in json_file if line.strip()]
    return []

def load_data(model):
    model_dir = f"../../../{model}_safim"
    data = defaultdict(list)
    for task in [API, BLOCK, CONTROL]:
        outputs = load_jsonl_if_exists(f"{model_dir}/{task}/{model}_safim_{task}_reasoning.jsonl")
        results = load_jsonl_if_exists(f"{model_dir}/{task}/{model}_safim_{task}_results.jsonl")
        assert len(outputs) == len(results)
        for i in range(len(outputs)):            
            assert outputs[i]["task_id"] == results[i]["task_id"]
            out = {
                **outputs[i],
                **results[i],
            }

            if out["passed"]:
                out["state"] = "success"
            else:
                if type(out["result"]) is not list:
                    results_lst = [{"exec_outcome": out["result"]}]
                else:
                    results_lst = out["result"]
                for res in results_lst:
                    if res["exec_outcome"] == "COMPILATION_ERROR":
                        out["state"] = "compile_failed"
                        break
                    elif res["exec_outcome"] == "RUNTIME_ERROR" or res["exec_outcome"] == "TIME_LIMIT_EXCEEDED":
                        out["state"] = "runtime_failed"
                        break
                    elif res["exec_outcome"] == "WRONG_ANSWER" or res["exec_outcome"] == "EMPTY":
                        out["state"] = "test_failed"
                        break
            
            assert "state" in out, out["result"]

            data[task].append(out)

    return data

def classify_trace(reasoning):
    prompt = PROMPT_TEMPLATE.format(reasoning=reasoning) 
    try: 
        response = model.generate_content(prompt) 
        content = response.text.strip() 
        json_start = content.find("{") 
        json_end = content.rfind("}") + 1 
        result = json.loads(content[json_start:json_end]) 
        return result 
    except Exception as e: 
        return { "reasoning_strategies": [], "notes": f"Failed to parse or generate: {str(e)}" }

def run_analysis(data, model, task):
    
    output_path = f"{model}_safim_{task}_analysis.jsonl"
    enriched = load_jsonl_if_exists(output_path)
    completed_tasks = set([item["task_id"] for item in enriched]) if len(enriched) > 0 else set()

    print(f"Running analysis for {task} on {model}")

    n = len(data[task])
    for i in tqdm(range(n), mininterval=0, miniters=1):

        item = data[task][i]
        task_id = item["task_id"]
        if task_id in completed_tasks:
            continue
        
        reasoning = item.get("reasoning", "")
        state = item.get("state")

        if not reasoning.strip():
            enriched_item = {
                **item,
                "reasoning_strategies": [],
                "notes": "No reasoning provided",
                "state": state,
            }
            continue

        result = classify_trace(reasoning)
        enriched_item = {
            **item,
            **result,
            "state": state,
        }

        enriched_item["reasoning_token_length"] = count_tokens(reasoning, model)

        enriched.append(enriched_item)

        if i % 5 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                for item in enriched:
                    f.write(json.dumps(item) + "\n")

        time.sleep(0.5) 

    with open(output_path, "w", encoding="utf-8") as f:
        for item in enriched:
            f.write(json.dumps(item) + "\n")

def process_results(model, task):
    path = f"{model}_safim_{task}_analysis.jsonl"
    data = load_jsonl_if_exists(path)

    records = []
    for item in data:
        strategies = item.get("reasoning_strategies", [])
        for strat in strategies:
            records.append({
                "question_id": item["task_id"],
                "strategy_type": strat["type"],
                "occurrences": strat["occurrences"],
                "proportion": strat["proportion"],
                "state": item.get("state"),
                "reasoning_token_length": item["reasoning_token_length"],
                "passed": item["passed"],
                "score": 1 if item["passed"] else 0
            })

    df = pd.DataFrame(records)
    return df

def analyze_occurance_rate(df):
    occurrence_summary = df.groupby(["strategy_type", "state"]).agg({
        "occurrences": "mean",
        "proportion": "mean",
        "score": ["mean", "count"]
    }).reset_index()

    occurrence_summary.columns = ["strategy", "state", "avg_occurrences", "avg_proportion", "avg_score", "count"]
    print(occurrence_summary)

    states = ["success", "compile_failed", "runtime_failed", "test_failed"]
    palette = sns.color_palette("tab10", n_colors=len(states))
    state_color_mapping = dict(zip(states, palette))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=occurrence_summary, x="avg_occurrences", y="strategy", hue="state", palette=state_color_mapping)
    plt.title("Avg Occurrences per Strategy by State")
    plt.tight_layout()
    plt.show()

def analyze_rates(df):
    patterns = (
        df.groupby(["state", "strategy_type"])["question_id"]
        .count()
        .unstack(fill_value=0)
        .T  
    )
    patterns["total"] = patterns.sum(axis=1)
    patterns["test_failure_rate"] = patterns.get("test_failed", 0) / patterns["total"]
    patterns["runtime_failure_rate"] = patterns.get("runtime_failed", 0) / patterns["total"]
    patterns["compile_failure_rate"] = patterns.get("compile_failed", 0) / patterns["total"]
    patterns["success_rate"] = patterns.get("success", 0) / patterns["total"]
    
    patterns = patterns.sort_values("success_rate", ascending=False)
    print(patterns[["test_failure_rate", "runtime_failure_rate", "compile_failure_rate", "success_rate"]])

def generate_analysis(model):
    data = load_data(model)
    for task in [API, BLOCK, CONTROL]:
        run_analysis(data, model, task)
        print()

def process_analysis(model, task):
    df = process_results(model, task)

    analyze_occurance_rate(df)
    print()
    analyze_rates(df)

def main():
    print()
    #generate_analysis(DEEPSEEK_R1)
    #process_analysis(DEEPSEEK_R1, BLOCK)

if __name__ == "__main__":
    main()
    