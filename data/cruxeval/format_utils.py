"""
CruxEval Format Utilities

Utilities for formatting compositions into batch requests and evaluating results.
"""

import json
import re
import argparse


def build_samples(input_file: str, num_funcs: int, output_file: str, model: str = "deepseek-ai/DeepSeek-R1"):
    """
    Build batch request samples from composition file.
    
    Args:
        input_file: Path to compositions JSONL file
        num_funcs: Number of functions in each composition
        output_file: Output path for batch requests
        model: Model identifier for API requests
    """
    data = []
    pre_prompt = """Based on the given Python code, which may contain errors, complete the assert statement with the 
output when executing the code on the given test case. Do not output any extra information, 
even if the function is incorrect or incomplete.

"""
    suf_prompt = """Only return the output of the function without any other information and assert statement. 
If the output is a string, enclose it in single quotes."""

    prompt_len = 0
    with open(input_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            qid = "request-"
            prompt = pre_prompt
            req = sample["input"]
            
            for i in range(num_funcs):
                func_id = sample[f"f{i+1}"]
                code = sample[f"code_f{i+1}"]
                qid += func_id + "_"
                prompt += f"#f{i+1}\n" + code + "\n\n"
                req = f"f{i+1}(" + req + ")"
            
            prompt += "assert " + req + " == \n" + suf_prompt
            prompt_len += len(prompt)
            
            entry = {
                "custom_id": qid[:-1],
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "max_tokens": 20000,
                },
                "ground_truth": sample["output"]
            }
            data.append(entry)
    
    with open(output_file, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    
    print(f"Generated {len(data)} batch requests")
    print(f"Average input length: {prompt_len / len(data):.0f} characters")


def extract_reasoning(batch_input_file: str, batch_result_file: str, output_file: str):
    """
    Extract reasoning and answers from batch results.
    
    Args:
        batch_input_file: Original batch input file with ground truth
        batch_result_file: Batch result file with model outputs
        output_file: Output path for extracted results
    """
    ground_truth = {}
    total = 0
    
    with open(batch_input_file, "r") as f:
        for line in f:
            total += 1
            data = json.loads(line)
            ground_truth[data["custom_id"]] = data["ground_truth"]
    
    results = {}
    total_score = 0
    
    with open(batch_result_file, "r") as f:
        for line in f:
            item = json.loads(line)
            request_id = item["custom_id"]
            composition_id = request_id[8:]  # Remove "request-" prefix
            content = item["response"]["body"]["choices"][0]["message"]["content"]

            if "</think>" not in content:
                results[composition_id] = {"reasoning": content, "answer": None, "score": 0}
                continue
            
            parts = re.split(r"</?think>", content)
            reasoning = parts[1].strip()
            answer = parts[2].strip()
            score = 1 if answer == ground_truth[request_id] else 0
            total_score += score
            results[composition_id] = {"reasoning": reasoning, "answer": answer, "score": score}

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"pass@1 score: {total_score / total:.4f}, total: {total}")


def change_batch_model(batch_input_file: str, model: str, output_file: str):
    """Change the model in batch request file."""
    output = []
    with open(batch_input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            data["body"]["model"] = model
            output.append(data)

    with open(output_file, "w") as f:
        for item in output:
            json.dump(item, f)
            f.write('\n')
    
    print(f"Updated {len(output)} requests to use model: {model}")


def evaluate(input_file: str):
    """Evaluate results from extracted reasoning file."""
    with open(input_file, "r") as f:
        data = json.load(f)
    
    total = len(data)
    score = sum(v["score"] for v in data.values())
    
    print(f"Score: {score}/{total} = {score/total:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CruxEval format utilities")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Build samples command
    build_parser = subparsers.add_parser("build", help="Build batch request samples")
    build_parser.add_argument("--input", "-i", required=True, help="Compositions JSONL file")
    build_parser.add_argument("--output", "-o", required=True, help="Output batch requests file")
    build_parser.add_argument("--num-funcs", "-n", type=int, required=True, help="Number of functions per composition")
    build_parser.add_argument("--model", "-m", default="deepseek-ai/DeepSeek-R1", help="Model identifier")
    
    # Extract reasoning command
    extract_parser = subparsers.add_parser("extract", help="Extract reasoning from batch results")
    extract_parser.add_argument("--input", "-i", required=True, help="Original batch input file")
    extract_parser.add_argument("--results", "-r", required=True, help="Batch results file")
    extract_parser.add_argument("--output", "-o", required=True, help="Output extracted results file")
    
    # Change model command
    model_parser = subparsers.add_parser("change-model", help="Change model in batch file")
    model_parser.add_argument("--input", "-i", required=True, help="Input batch file")
    model_parser.add_argument("--model", "-m", required=True, help="New model identifier")
    model_parser.add_argument("--output", "-o", required=True, help="Output batch file")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate results")
    eval_parser.add_argument("--input", "-i", required=True, help="Extracted results file")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_samples(args.input, args.num_funcs, args.output, args.model)
    elif args.command == "extract":
        extract_reasoning(args.input, args.results, args.output)
    elif args.command == "change-model":
        change_batch_model(args.input, args.model, args.output)
    elif args.command == "evaluate":
        evaluate(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
