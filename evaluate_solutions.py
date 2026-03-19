import argparse
import jsonlines
import shlex
import subprocess
import sys
import tempfile

from pathlib import Path

def evaluate_problem(problem: dict, output_path: Path):
    benchmark = problem["benchmark"]
    optimized_code = problem["optimized_code"]
    DATASET_SIZE = "MEDIUM"

    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{benchmark}.json"

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as f:
        f.write(optimized_code)
        f.seek(0)

        run_eval_args = (
            f"uv run evaluate-llm-kernel.py --benchmark {benchmark} "
            f"--submission-file {shlex.quote(f.name)} --dataset-size {DATASET_SIZE} "
            f"--warmups 1 --runs 3 --json-out {shlex.quote(str(output_file))}"
        )
        # print(run_eval_args)
        res = subprocess.run(shlex.split(run_eval_args), text=True, capture_output=True)
        if res.returncode != 0:
            if output_file.exists():
                # print(
                #     f"Evaluation for {benchmark} returned a non-zero status, "
                #     f"but wrote {output_file}."
                # )
                # if res.stdout:
                #     print(res.stdout)
                # if res.stderr:
                #     print(res.stderr, file=sys.stderr)
                return

            raise RuntimeError(
                f"Evaluation failed for {benchmark} without producing an output file.\n"
                f"stdout:\n{res.stdout}\n"
                f"stderr:\n{res.stderr}"
            )
    

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
    args = parser.parse_args()

    problems = []
    with jsonlines.open(args.input) as reader:
        for obj in reader:
            problems.append(obj)

    for problem in problems:
        evaluate_problem(problem, args.output)

if __name__ == "__main__":
    main()
