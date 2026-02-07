import os
import json
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

# === CONFIGURATION ===
PROJECT_ID = "YOUR PROJECT ID HERE"
LOCATION = "us-central1"
client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="us-central1",
)
model = "gemini-2.0-flash"

WINDOW_SIZE = 5
DELAY = 0.5
MAX_RETRIES = 4
MAX_PARALLEL = 3

# === PROMPT ===
TREE_PROMPT_TEMPLATE = """
You are building a reasoning tree from segments of thought. Each segment expresses a part of a reasoning process.

Here are previous thoughts in the structure:
{tree_so_far}

Now consider this new segment:
```
{new_segment}
```

Decide where to attach this new segment in the tree. You must use one of the three relation types defined below.

Relation Categories
- Continuation: The new segment directly builds upon the parent's idea. It can do this by:
    1. Adding more details, explanations, or examples.
    2. Providing evidence or justification.
    3. Refining or clarifying the parent's point.
- Contrast: The new segment proposes a different, alternative, or opposing idea compared to the parent node.
- Rephrase: The new segment expresses the exact same core idea as the parent but in different words.

Return your answer in this JSON format:
```json
{{
  "parent_id": "thought_X",  // id of parent thought, or "root" for top-level
  "relation": "Continuation" | "Contrast" | "Rephrase"
}}
"""

def load_labeled_segments(filepath):
    with open(filepath) as f:
        return json.load(f)

def format_tree_structure(nodes):
    def format_node(n, depth=0):
        indent = "  " * depth
        return f"{indent}- [{n['id']}] {n['text']} ({n['label']}) relation: {n['relation']}"

    id_to_node = {n['id']: n for n in nodes}
    parent_to_children = {}
    for n in nodes:
        parent = n.get("parent", "root")
        parent_to_children.setdefault(parent, []).append(n)

    def build_tree_str(parent_id="root", depth=0):
        lines = []
        for child in parent_to_children.get(parent_id, []):
            lines.append(format_node(child, depth))
            lines.extend(build_tree_str(child["id"], depth + 1))
        return lines

    return "\n".join(build_tree_str())

def ask_model_for_attachment(tree_nodes, new_segment):
    prompt = TREE_PROMPT_TEMPLATE.format(
        tree_so_far=format_tree_structure(tree_nodes),
        new_segment=new_segment["text"]
    )
    all_ids = [n["id"] for n in tree_nodes]
    all_ids.append("root")
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
            parsed = json.loads(content[json_start:json_end])
            parent_id = parsed.get("parent_id")
            relation = parsed.get("relation")
            if not parent_id or parent_id not in all_ids or relation not in {"Continuation", "Contrast", "Rephrase"}:
                raise ValueError("Invalid structure returned")
            return parsed
        except Exception as e:
            print(e)
            if attempt == MAX_RETRIES - 1:
                return {"parent_id": "root", "relation": "elaboration", "error": str(e)}
            time.sleep(1)

def build_tree_for_file(file_path, output_dir):
    with open(file_path) as f:
        item = json.load(f)
    qid = Path(file_path).stem.replace("qid_", "")
    output_path = os.path.join(output_dir, f"qid_{qid}.json")
    if os.path.exists(output_path):
        print(f"Skipping {qid} (already exists)")
        return

    segments = item.get("segments", [])
    id_to_node = {}
    root = {"id": "root", "children": [], "metadata": item.get("metadata", {}), "original_reasoning": item.get("original_reasoning", "")}

    for idx, seg in enumerate(segments):
        seg_id = f"thought_{idx}"
        parent_info = ask_model_for_attachment(list(id_to_node.values()), seg) if id_to_node else {"parent_id": "root", "relation": "elaboration"}
        node = {
            "id": seg_id,
            "text": seg["text"],
            "label": seg.get("label", "unknown"),
            "relation": parent_info.get("relation", "elaboration"),
            "children": [],
            "parent_id": parent_info.get("parent_id", "root"),
        }
        id_to_node[seg_id] = node

        parent_id = parent_info.get("parent_id", "root")
        parent_node = root if parent_id == "root" else id_to_node.get(parent_id)
        parent_node["children"].append(node)

        time.sleep(DELAY)

    with open(output_path, "w") as f:
        json.dump({"qid": qid, "tree": root}, f, indent=2)
    print(f"Saved Tree for QID {qid}")

def tree_batch(input_dir: str, output_dir: str, limit: int = None):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(Path(input_dir).glob("qid_*.json"))
    files = files[:limit] if limit else files

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [executor.submit(build_tree_for_file, file, output_dir) for file in files]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    label_dir_tree_dir_tuple = [
        # add your (label_dir, tree_dir) pairs here
    ]
    for label_dir, tree_dir in label_dir_tree_dir_tuple:
        tree_batch(label_dir, tree_dir)