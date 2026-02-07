# Data Generation

This directory contains tools for generating multi-step code reasoning compositions from CruxEval and SAFIM benchmarks.

## Directory Structure

```
data/
├── cruxeval/           # CruxEval function chaining compositions
│   ├── generate_composition.py    # Main composition generator
│   ├── format_utils.py            # Batch formatting utilities
│   └── requirements.txt           # Python dependencies
│
└── safim/              # SAFIM code completion compositions
    ├── generate_composition.py    # Main composition generator
    ├── check_results.py           # Result verification
    ├── data_utils.py              # Dataset loading utilities
    ├── exec_utils.py              # Code execution utilities
    ├── build.sh                   # Environment setup script
    ├── start_docker.sh            # ExecEval service launcher
    ├── requirements.txt           # Python dependencies
    └── ExecEval/                  # Docker-based code execution service
```

## CruxEval Compositions

Generates chained function compositions where the output of one function feeds into the next.

### Prerequisites

- Python 3.8+
- CruxEval dataset (download from HuggingFace or original source)

### Usage

```bash
cd cruxeval

# Generate compositions with 3 chained functions
python generate_composition.py \
    --input /path/to/cruxeval.jsonl \
    --output level3_compositions.jsonl \
    --num-funcs 3 \
    --max-samples 200 \
    --seed 42

# Build batch requests for API evaluation
python format_utils.py build \
    --input level3_compositions.jsonl \
    --output batch_requests.jsonl \
    --num-funcs 3 \
    --model "deepseek-ai/DeepSeek-R1"

# Extract results from batch output
python format_utils.py extract \
    --input batch_requests.jsonl \
    --results batch_output.jsonl \
    --output results.json

# Evaluate results
python format_utils.py evaluate --input results.json
```

### Command Line Arguments

**generate_composition.py:**
- `--input, -i`: Path to CruxEval JSONL file (required)
- `--output, -o`: Output file path (required)
- `--num-funcs, -n`: Number of functions per chain (default: 3)
- `--max-samples, -m`: Maximum compositions to generate (default: 200)
- `--timeout, -t`: Timeout per function in seconds (default: 10)
- `--seed, -s`: Random seed for reproducibility

## SAFIM Compositions

Generates chained code completion problems where test outputs propagate through the chain.

### Prerequisites

- Python 3.8+
- Docker installed and running
- Ubuntu x86 system (for ExecEval Docker)

### Setup

```bash
cd safim

# Install dependencies and build Docker image
./build.sh
```

### Usage

```bash
# Terminal 1: Start the ExecEval service
./start_docker.sh

# Terminal 2: Generate compositions
python generate_composition.py 2 200 --seed 42   # Level 2, 200 samples
python generate_composition.py 3 200 --seed 42   # Level 3, 200 samples

# Check model-generated results
python check_results.py model_output.json results.json
```

### Command Line Arguments

**generate_composition.py:**
- `level`: Composition level (number of chained completions)
- `sample_size`: Number of compositions to generate
- `--port`: ExecEval server port (default: 5000)
- `--output, -o`: Output file path
- `--seed, -s`: Random seed for reproducibility

**check_results.py:**
- `input_path`: Path to model output JSON file
- `output_path`: Path to save evaluation results
- `--port`: ExecEval server port (default: 5000)

## Output Formats

### CruxEval Composition Format

```json
{
  "f1": "sample_id_1",
  "f2": "sample_id_2", 
  "f3": "sample_id_3",
  "code_f1": "def f(x): ...",
  "code_f2": "def f(x): ...",
  "code_f3": "def f(x): ...",
  "input": "initial_input",
  "output": "final_output"
}
```

### SAFIM Composition Format

```json
{
  "eval_prompt1": "code_with_placeholder",
  "ground_truth1": "correct_completion",
  "task_id1": "safim_task_id",
  "eval_prompt2": "...",
  "ground_truth2": "...",
  "task_id2": "...",
  "unit_tests": [{"input": "...", "output": ["..."]}]
}
```

## Notes

- Composition generation involves random sampling; use `--seed` for reproducibility
- ExecEval service must be running before generating SAFIM compositions
- Generated compositions are verified for executability during generation
