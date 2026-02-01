"""
SAFIM Composition Generator

Generates chained code completion compositions from SAFIM dataset for evaluating
reasoning models on multi-step code completion tasks.
"""

import argparse
import json
import random

from data_utils import load_dataset
from exec_utils import build_execeval, run_combine_test


def main():
    parser = argparse.ArgumentParser(description="Generate SAFIM compositions")
    parser.add_argument("level", type=int, help="Composition level (number of chained completions)")
    parser.add_argument("sample_size", type=int, help="Number of compositions to generate")
    parser.add_argument("--port", type=int, default=5000, help="ExecEval server port (default: 5000)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path (default: safim_level<N>_compositions.jsonl)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    build_execeval(args)

    output_path = args.output or f"safim_level{args.level}_compositions.jsonl"

    # Load Python block completion problems from SAFIM
    problems = []
    for p in load_dataset('block'):
        if p["lang"] == "python":
            problems.append(p)
    
    print(f"Loaded {len(problems)} Python block completion problems")

    results = []

    while len(results) < args.sample_size:
        chain = []
        
        while len(chain) < args.level:
            chain = []
            chain.append(random.sample(problems, 1)[0])
            unit_tests = chain[0]['unit_tests']
            
            for i in range(args.level - 1):
                passed = False
                sample_problem = None
                new_unit_tests = None
                try_times = 0
                
                while not passed and try_times < 100:
                    sample_problem = random.sample(problems, 1)[0]
                    new_unit_tests, passed = run_combine_test(unit_tests, sample_problem)
                    try_times += 1
                
                if not passed:
                    break
                
                chain.append(sample_problem)
                unit_tests = new_unit_tests

        if len(chain) < args.level:
            continue
        
        result = {}
        for idx, p in enumerate(chain):
            result[f"eval_prompt{idx + 1}"] = p["eval_prompt"]
            result[f"ground_truth{idx + 1}"] = p["ground_truth"]
            result[f"task_id{idx + 1}"] = p["task_id"]
        result["unit_tests"] = unit_tests
        
        results.append(result)
        print(f"Generated composition {len(results)}/{args.sample_size}")
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    print(f"Done! Generated {len(results)} compositions to {output_path}")


if __name__ == '__main__':
    main()
