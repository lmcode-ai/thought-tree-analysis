"""
CruxEval Composition Generator

Generates chained function compositions from CruxEval dataset for evaluating
reasoning models on multi-step code execution tasks.
"""

import ast
import json
import textwrap
import signal
import random
import argparse


def load_samples(json_path: str):
    """Load samples from a JSONL file."""
    samples = []
    with open(json_path, 'r') as file:
        for line in file:
            samples.append(json.loads(line))
    return samples


def safe_eval(s: str):
    """Safely evaluate a string as a Python literal."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("Function call timed out")


def execute_with_timeout(func, args=(), timeout_seconds=60):
    """
    Run func(*args) but raise TimeoutError if it takes longer than timeout_seconds.
    Uses SIGALRM, so only works on Unix & only in the main thread.
    """
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return func(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def annotate_samples(samples: list):
    """
    Parse input/output strings into Python objects, record types,
    and bind each code snippet to a callable function.
    """
    for s in samples:
        s['input_obj'] = safe_eval(s['input'])
        s['output_obj'] = safe_eval(s['output'])
        
        inp = s['input_obj']
        s['input_args'] = inp if isinstance(inp, tuple) else (inp,)
        
        ns = {}
        exec(textwrap.dedent(s['code']), ns)
        s['func'] = ns['f']
        
        s['type_info'] = {
            'input_types': tuple(type(arg).__name__ for arg in s['input_args']),
            'output_type': type(s['output_obj']).__name__
        }


def type_match(out, sample):
    """Check if output type matches sample's expected input type."""
    if len(sample['input_args']) != 1:
        return False
    
    inp = sample['input_args'][0]

    if not isinstance(out, type(inp)):
        return False
    
    if isinstance(out, (list, tuple)) and len(out) > 0:
        if len(inp) == 0:
            return False
        elem_t = type(inp[0])
        return all(isinstance(e, elem_t) for e in out)
    
    return True


def find_chained_compositions(samples: list, outfile, num_funcs=4, timeout_seconds=10, max_samples=200):
    """
    Randomly builds one chain per starting sample, avoiding repeats in a chain.
    Stops after max_samples chains.
    """
    count = 0

    while count < max_samples:
        s0 = random.choice(samples)
        chain_samples = [s0]
        current_output = s0['output_obj']

        print(f"[f1] s1 id={s0['id']}, {count} chains generated so far")

        for depth in range(1, num_funcs):
            candidates = [s for s in samples if s not in chain_samples and type_match(current_output, s)]
            if not candidates:
                break

            s_next = random.choice(candidates)
            try:
                current_output = execute_with_timeout(s_next['func'], (current_output,), timeout_seconds=timeout_seconds)
            except TimeoutError:
                print(f"{'  '*depth}[f{depth+1}] id={s_next['id']} timed out")
                break
            except Exception:
                break

            chain_samples.append(s_next)
            print(f"{'  '*depth}[f{depth+1}] {chain_samples[0]['id']}→...→{s_next['id']} OK")

        if len(chain_samples) == num_funcs:
            record = {
                **{f"f{i+1}": chain_samples[i]['id'] for i in range(num_funcs)},
                **{f"code_f{i+1}": chain_samples[i]['code'] for i in range(num_funcs)},
                "input": chain_samples[0]['input'],
                "output": repr(current_output),
            }
            outfile.write(json.dumps(record) + "\n")
            outfile.flush()
            count += 1


def main():
    parser = argparse.ArgumentParser(description="Generate chained function compositions from CruxEval")
    parser.add_argument("--input", "-i", required=True, help="Path to CruxEval JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--num-funcs", "-n", type=int, default=3, help="Number of functions per chain (default: 3)")
    parser.add_argument("--max-samples", "-m", type=int, default=200, help="Maximum number of compositions to generate (default: 200)")
    parser.add_argument("--timeout", "-t", type=int, default=10, help="Timeout per function execution in seconds (default: 10)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Loading samples from {args.input}...")
    samples = load_samples(args.input)
    random.shuffle(samples)
    
    print(f"Annotating {len(samples)} samples...")
    annotate_samples(samples)

    print(f"Generating {args.max_samples} compositions with {args.num_funcs} functions each...")
    with open(args.output, "w") as out:
        find_chained_compositions(
            samples, out,
            num_funcs=args.num_funcs,
            timeout_seconds=args.timeout,
            max_samples=args.max_samples
        )
    
    print(f"Done! Output written to {args.output}")


if __name__ == "__main__":
    main()
