import argparse
import json
import jsonlines
import shlex
import subprocess
import tempfile

from pathlib import Path

def evaluate_problem(problem: dict, output_path: Path):
    benchmark = problem["benchmark"]
    optimized_code = problem["optimized_code"]
    DATASET_SIZE = "MEDIUM"

    output_file = output_path / f"{benchmark}.json"

    with tempfile.NamedTemporaryFile(suffix=".py") as f:
        f.write(optimized_code)
        f.seek(0)

        run_eval_args = f"python evaluate-llm-kernel.py --benchmark {benchmark} --submission-file {f.name} --dataset-size {DATASET_SIZE} --warmups 1 --runs 3 --json-out {output_file}"
        print(run_eval_args)
        res = subprocess.run(shlex.split(run_eval_args), text=True, capture_output=True)
        assert res.returncode == 0, f"{res.stdout}\n\n{res.stderr}"
    

def main():
    """Parse command line arguments and generate solutions for enamel benchmark."""
    parser = argparse.ArgumentParser(description="Generate solutions for enamel benchmark")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input JSONlines folder"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file path for generations (will be List[List[str]] format)"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model name for generation"
    )
    args = parser.parse_args()

    problems = []
    with jsonlines.open(args.input) as reader:
        for obj in reader:
            problems.append(obj)

    for problem in problems:
        evaluate_problem(problem, args.output)

if __name__ == "__main__":
    main()
