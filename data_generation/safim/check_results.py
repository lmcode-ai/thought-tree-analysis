"""
SAFIM Results Checker

Evaluates model-generated code completions against unit tests.
"""

import argparse
import json

from data_utils import load_dataset
from exec_utils import build_execeval, run_test


def main():
    parser = argparse.ArgumentParser(description="Check SAFIM composition results")
    parser.add_argument("input_path", type=str, help="Path to model output JSON file")
    parser.add_argument("output_path", type=str, help="Path to save results")
    parser.add_argument("--port", type=int, default=5000, help="ExecEval server port (default: 5000)")
    args = parser.parse_args()

    build_execeval(args)

    # Load problem database
    problems = {}
    for p in load_dataset('block'):
        problems[p['task_id']] = p

    with open(args.input_path, 'r') as f:
        items = json.load(f)

    passed_count = 0
    result = {}

    for key, item in items.items():
        task_list = []
        code_list = []
        print(f"Evaluating: {key}")
        
        meta = item.get('metadata')
        idx = 1
        while True:
            task_key = f'task_id{idx}'
            code_key = f'code{idx}'
            if task_key in meta and code_key in item:
                task_list.append(meta.get(task_key))
                code_list.append(item.get(code_key))
                idx += 1
            else:
                break
        
        overall_passed = True
        pass_list = []
        status = "correct"

        for taskid, code in zip(task_list, code_list):
            problem = problems.get(taskid)
            if problem is None or code is None:
                overall_passed = False
                break
            
            completion = {'task_id': problem['task_id'], 'completion': code}
            test_result, passed = run_test(problem, completion)
            stat = test_result[0]['exec_outcome']
            
            if stat == "COMPILATION_ERROR" and status == "correct":
                status = "incorrect"
            if stat == "RUNTIME_ERROR" and status == "correct":
                status = "incorrect"
            if stat == "TIME_LIMIT_EXCEEDED" and status == "correct":
                status = "incorrect"
            if stat == "WRONG_ANSWER" and status == "correct":
                status = "incorrect"
            
            pass_list.append(passed)
            if not passed:
                overall_passed = False
        
        if status == "correct" and not overall_passed:
            status = "incomplete"
        
        print(f"  Status: {status}, Passed: {overall_passed}")
        result[key] = status
        
        if overall_passed:
            passed_count += 1

    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Passed: {passed_count}/{len(items)} ({100*passed_count/len(items):.2f}%)")
    print(f"Results saved to: {args.output_path}")


if __name__ == '__main__':
    main()
