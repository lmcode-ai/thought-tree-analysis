"""
SAFIM Execution Utilities

Utilities for code execution and testing via ExecEval service.
"""

from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import requests


class ExecOutcome(Enum):
    """Execution outcome categories."""
    PASSED = "PASSED"
    WRONG_ANSWER = "WRONG_ANSWER"
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"


@dataclass
class ExtendedUnittest:
    """Extended unittest with execution results."""
    input: str
    output: List[str] = field(default_factory=list)
    result: Optional[str] = None
    exec_outcome: Optional[ExecOutcome] = None

    def json(self):
        _json = self.__dict__
        if self.exec_outcome is not None:
            _json["exec_outcome"] = self.exec_outcome.name
        return _json

    @classmethod
    def from_json(cls, _json):
        return cls(
            input=_json.get("input", ""),
            output=_json.get("output", list()),
            result=_json.get("result", None),
            exec_outcome=_json.get("exec_outcome", None),
        )


class APICommunication:
    """Communication interface for ExecEval service."""
    _session: requests.Session

    def __init__(self, server_url: str = "http://localhost:5000"):
        self._session = requests.Session()
        self.execute_code_url = f"{server_url}/api/execute_code"
        self.get_runtimes_url = f"{server_url}/api/all_runtimes"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    def get_runtimes(self):
        return self._session.get(self.get_runtimes_url).json()

    def execute_code(
        self,
        language: str,
        source_code: str,
        unittests: List[dict],
        limits: Optional[dict] = None,
        block_network: bool = True,
        stop_on_first_fail: bool = True,
        use_sanitizer: bool = False,
        compiler_program_name: Optional[str] = None,
        compiler_flags: Optional[str] = None,
        interpreter_cmd: Optional[str] = None,
        interpreter_flags: Optional[str] = None,
        sample_id: Optional[int] = None,
        task_id: Union[str, int, None] = None,
    ) -> Tuple[List[ExtendedUnittest], Optional[int], Union[str, int, None]]:
        if language is None:
            raise ValueError("EmptyLanguage")
        if source_code is None:
            raise ValueError("EmptySourceCode")
        if unittests is None or len(unittests) == 0:
            raise ValueError("EmptyUnittest")

        request_body = dict(
            language=language,
            source_code=source_code,
            unittests=unittests,
            limits=limits if isinstance(limits, dict) else None,
            compile_cmd=compiler_program_name,
            compile_flags=compiler_flags,
            execute_cmd=interpreter_cmd,
            execute_flags=interpreter_flags,
            block_network=block_network,
            stop_on_first_fail=stop_on_first_fail,
            use_sanitizer=use_sanitizer,
        )
        
        try:
            json_response = self._session.post(
                self.execute_code_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
            ).json()
        except requests.exceptions.JSONDecodeError:
            json_response = {
                "task_id": task_id,
                "data": [{"exec_outcome": "COMPILATION_ERROR", "result": "", "passed": False}]
            }

        if "data" not in json_response:
            return json_response, sample_id, task_id

        return (json_response["data"], sample_id, task_id)


# Language to compiler mapping
LANG_TO_COMPILER = {
    "cpp": "GNU C++17",
    "csharp": "Mono C#",
    "java": "Java 17",
    "python": "PyPy 3"
}

# Global ExecEval instance
execeval: APICommunication = None


def run_test(problem, completion):
    """
    Run unit tests on a code completion.
    
    Args:
        problem: Problem dictionary with eval_prompt and unit_tests
        completion: Completion dictionary with task_id and completion code
        
    Returns:
        Tuple of (test_results, all_passed)
    """
    global execeval
    assert problem['task_id'] == completion['task_id']
    
    code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
    result = execeval.execute_code(
        LANG_TO_COMPILER[problem['lang']],
        code,
        problem['unit_tests'],
        task_id=problem['task_id']
    )[0]
    
    if not (isinstance(result, list) and isinstance(result[0], dict)):
        print(result)
        return "COMPILATION_ERROR", False
    
    for o in result:
        if o['result'] is not None and len(o['result']) > 1000:
            o['result'] = o['result'][:1000]
    
    return result, all(o['exec_outcome'] == 'PASSED' for o in result)


def run_combine_test(unit_tests, problem):
    """
    Run combination test to chain problems together.
    
    Args:
        unit_tests: Unit tests from previous problem
        problem: Next problem to chain
        
    Returns:
        Tuple of (new_unit_tests, success)
    """
    global execeval
    code = problem['eval_prompt'].replace("{{completion}}", problem['ground_truth'])

    new_unit_tests = []
    for test in unit_tests:
        new_unit_tests.append({
            'input': test['output'][0],
            'output': [""]
        })

    result = execeval.execute_code(
        LANG_TO_COMPILER[problem['lang']],
        code,
        new_unit_tests,
        task_id=problem['task_id']
    )[0]

    if not (isinstance(result, list) and isinstance(result[0], dict)):
        return "COMPILATION_ERROR", False

    final_unit_tests = []
    result_index = 0
    for test in unit_tests:
        outputs = [result[result_index]['result']]
        result_index += 1
        final_unit_tests.append({
            'input': test['input'],
            'output': outputs
        })

    passed = all(
        o['exec_outcome'] == 'PASSED' or o['exec_outcome'] == 'WRONG_ANSWER'
        for o in result
    ) and not all(o['exec_outcome'] == 'PASSED' for o in result)
    
    return final_unit_tests, passed


def build_execeval(args: Namespace):
    """Initialize the global ExecEval instance."""
    global execeval
    execeval = APICommunication(server_url=f"http://localhost:{args.port}")
