import argparse
import json
import jsonlines
import re
import time

from typing import List
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_code_blocks(response: str) -> str:
    """Extract code blocks from markdown-formatted response.

    Handles triple-backtick fenced blocks and prefers Python-like labels.

    Args:
        response: String containing markdown code blocks

    Returns:
        Concatenated extracted code blocks
    """
    if response is None:
        return ""

    python_labels = {
        "",
        "py",
        "py3",
        "python",
        "python3",
    }

    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    python_blocks = []
    unlabeled_blocks = []
    other_blocks = []

    for raw_label, raw_code in pattern.findall(response):
        code = raw_code.strip()
        if not code:
            continue

        label = raw_label.strip().split()[0].lower() if raw_label.strip() else ""
        if label in python_labels and label:
            python_blocks.append(code)
        elif not label:
            unlabeled_blocks.append(code)
        else:
            other_blocks.append(code)

    if python_blocks:
        return "\n".join(python_blocks)
    if unlabeled_blocks:
        return "\n".join(unlabeled_blocks)
    if other_blocks:
        return "\n".join(other_blocks)
    return ""

def build_code_opt_prompt(original_code: str) -> str:
    """Build an optimization prompt using a simple on-disk template with safe token replacement.

    The template must contain tokens {{LANGUAGE_NAME}}, {{SOURCE_FILENAME}}, {{BASELINE_SECONDS}},
    and markers :::HEADER::: and :::CODE::: which will be replaced verbatim.
    Falls back to an inline template if the file is missing.
    """

    return (
        f"You are an expert python performance engineer.\n\n"
        # f"Python version: 3.10, Numpy version 1.26.4\n"
        f"Python version: 3.10"
        f"Task: Optimize the provided function for speed while preserving exact behavior and I/O.\n"
        f"- Do not change the function signature expected by the harness.\n"
        f"- Provide a full replacement for the function.\n"
        f"- Return only the code in a single fenced block.\n\n"
        f"--- current source ---\n{original_code}\n"
    )

def llm_generate(client, model: str, prompt: str) -> str:
    completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
            # temperature=0.6,
            # top_p=0.95,
            temperature=1.0,
            top_p=1.0,
        )

    text = completion.choices[0].message.content
    return text

def generate_solutions(problems: List[dict], model: str) -> List[dict]:
    """Generate solutions for a list of problems.

    Args:
        problems: List of problem dictionaries with 'prompt' field
        model: Model name to use for generation
        num_completions: Number of solutions to generate per problem

    Returns:
        List[List[str]]: List of lists where result[i] contains solutions for problem i
    """
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="ryan123",
    )

    # Initialize result list with empty lists for each problem
    generations : List[dict] = [{} for _ in range(len(problems))]
    num_none = 0

    start = time.time()

    def process_problem(problem: dict, idx: int) -> tuple[str | None, int]:
        baseline_code = problem["src_code"]
        prompt = build_code_opt_prompt(baseline_code)
        response = llm_generate(client, model, prompt)
        code = extract_code_blocks(response)
        return code, idx

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_problem, problem, idx)
                   for idx, problem in enumerate(problems)]
        for future in as_completed(futures):
            solution, idx = future.result()
            problem = problems[idx]
            generations[idx] = {"benchmark" : problem["benchmark"], "src_code" : problem["src_code"], "optimized_code" : solution}
            if not solution:
                num_none += 1

    end = time.time()

    print(f"Problems with no solutions: {num_none}/{len(problems)}, time: {end - start:.2f}s")
    return generations

def main():
    """Parse command line arguments and generate solutions for enamel benchmark."""
    parser = argparse.ArgumentParser(description="Generate solutions for enamel benchmark")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input JSONlines path"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
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

    # Generate solutions: returns List[List[str]]
    generations = generate_solutions(problems, args.model)

    with jsonlines.open(args.output, "w") as writer:
        writer.write_all(generations)

    print(f"Saved {len(generations)} problems to {args.output}")

if __name__ == "__main__":
    main()
