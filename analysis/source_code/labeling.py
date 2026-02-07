import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from google import genai
from google.genai import types

# === CONFIGURATION ===
PROJECT_ID = "YOUR PROJECT ID HERE"      
LOCATION = "us-central1" 
MAX_PARALLEL = 5
DELAY = 0.5       # Delay to avoid rate limit
MAX_RETRIES = 4

# vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="us-central1",
)
model = "gemini-2.0-flash"

# === PROMPT TEMPLATE ===
SEGMENT_PROMPT = """
You are an expert in reasoning trace analysis.

Classify the following reasoning segment into one of these strategies:

- "code_analysis": actively analyzing what the original code does
- "mental_execution": simulating input/output behavior
- "test_generation": proposing test cases or edge conditions
- "bug_fixing": describing or fixing syntax/logic errors
- "language_mapping": translating code constructs across languages
- "high_level_plan": outlining the approach or plan without executing it
- "empty": no meaningful reasoning

Only return a JSON like:
```json
{{"type": "..."}}
Segment:
{segment}
"""

MULTISHOT_SEGMENT_PROMPT = """
You are an expert in reasoning trace analysis.

Classify the following reasoning segment into one of these strategies:

- "code_analysis": analyzing what the original code does
- "mental_execution": simulating input/output behavior
- "test_generation": proposing test cases or edge conditions
- "bug_fixing": describing or fixing syntax/logic errors
- "language_mapping": translating code constructs across languages
- "high_level_plan": outlining the approach or plan
- "empty": no meaningful reasoning

Only return a JSON like:
```json
{{"type": "..."}}

Examples:
---
Segment:
"The Java code reads an integer n and stores values in an array. It uses a loop to iterate."
{{ "type": "code_analysis" }}

Segment:
"Okay, let's simulate what happens if n = 5 and a = 2."
{{ "type": "mental_execution" }}

Segment:
"We should test when the array is empty or when the inputs are negative."
{{ "type": "test_generation" }}
---

Segment:
{segment}
"""


def classify_segment(segment: str, use_few_shot: bool = False) -> Dict:
    prompt = (
        MULTISHOT_SEGMENT_PROMPT if use_few_shot else SEGMENT_PROMPT
    ).format(segment=segment.strip())
    
    for attempt in range(MAX_RETRIES):
        try:
            # response = model.generate_content(prompt)
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                ),
            )
            content = response.text.strip()
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            print(f"one labeling done: {content}")
            return json.loads(content[json_start:json_end])
        except Exception as e:
            print(e)
            if attempt == MAX_RETRIES - 1:
                return {"type": "error", "error": str(e)}
            time.sleep(1)


def label_single_qid(qid, item, out_dir):
    output_path = os.path.join(out_dir, f"qid_{qid}.json")
    if os.path.exists(output_path):
        print(f"↪Skipping {qid} (already exists)")
        return

    segments = item.get("segments", [])
    labeled = []

    for seg in segments:
        result = classify_segment(seg["text"])
        labeled.append({
            **seg,
            "label": result.get("type", "error"),
            "label_info": result
        })
        time.sleep(DELAY)

    output_data = {
        "segments": labeled,
        "metadata": item.get("metadata", {}),
        "original_reasoning": item.get("original_reasoning", "")
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved QID: {qid}")


def label_segments_for_file(input_path: str, output_dir: str, limit: int = None):
    with open(input_path) as f:
        data = json.load(f)

    if limit:
        data = dict(list(data.items())[:limit])

    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [
            executor.submit(label_single_qid, qid, item, output_dir)
            for qid, item in data.items()
        ]
        for future in as_completed(futures):
            future.result()


def batch_process(input_files: List[str], output_dir: str, limit=None):
    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    for input_path in input_files:
        name = Path(input_path).stem + "_labeled"
        output_path = os.path.join(output_dir, name)
        label_segments_for_file(input_path, output_path, limit)

if __name__ == "__main__":
    input_files = [
        # add your input files here, e.g. "../{dataset}/segment/qwq_incorrect_level2_to_level3.json"
    ]
    output_dir = "../{dataset}/segment_labeled"
    batch_process(input_files, output_dir)
