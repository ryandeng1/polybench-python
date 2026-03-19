from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import types
import uuid
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from benchmarks import benchmark_classes
from benchmarks.polybench_classes import ArrayImplementation, DataSetSize
from benchmarks.polybench_classes import PolyBenchOptions, PolyBenchSpec, PolyBenchSpecFile


class EvaluationError(RuntimeError):
    pass


class InvalidSubmission(EvaluationError):
    pass


class UnknownBenchmark(EvaluationError):
    pass


_ALLOWED_NON_STDLIB_IMPORTS = {"numpy"}


def load_submission_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def list_benchmark_ids() -> list[str]:
    ids = set()
    for benchmark_class in benchmark_classes:
        ids.add(benchmark_class.__module__.split(".")[-1])
    return sorted(ids)


def evaluate_kernel(
    benchmark_id: str,
    submission_source: str,
    dataset_size: str = "LARGE",
    array_implementation: str = "numpy",
    warmups: int = 1,
    runs: int = 5,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict:
    result = {
        "benchmark": _canonical_benchmark_id(benchmark_id),
        "array_implementation": str(array_implementation).lower(),
        "correct": False,
        "status": "error",
        "baseline_median_ms": None,
        "candidate_median_ms": None,
        "speedup": None,
        "baseline_samples_ms": [],
        "candidate_samples_ms": [],
        "error": None,
    }

    try:
        normalized_dataset = _normalize_dataset_size(dataset_size)
        normalized_array_implementation = _normalize_array_implementation(array_implementation)
        _validate_positive_int("warmups", warmups, allow_zero=True)
        _validate_positive_int("runs", runs, allow_zero=False)

        target_class, benchmark_spec = _resolve_benchmark(benchmark_id)
        prototype = _instantiate_benchmark(
            target_class,
            benchmark_spec,
            normalized_dataset,
            normalized_array_implementation,
        )
        target_signature = _signature_shape(prototype.__class__.kernel)

        submission_kernel = _load_submission_kernel(submission_source)
        if _signature_shape(submission_kernel) != target_signature:
            raise InvalidSubmission(
                f'Submission kernel signature does not match benchmark "{result["benchmark"]}". '
                f'Expected {_format_signature(prototype.__class__.kernel)}, '
                f"received {_format_signature(submission_kernel)}."
            )

        baseline_outputs, _ = _execute_once(
            target_class,
            benchmark_spec,
            normalized_dataset,
            normalized_array_implementation,
        )
        candidate_outputs, _ = _execute_once(
            target_class,
            benchmark_spec,
            normalized_dataset,
            normalized_array_implementation,
            kernel_override=submission_kernel,
        )

        outputs_match, mismatch_reason = _compare_outputs(
            baseline_outputs,
            candidate_outputs,
            rtol=rtol,
            atol=atol,
        )
        if not outputs_match:
            result["status"] = "incorrect"
            result["error"] = mismatch_reason
            return result

        baseline_samples = _collect_samples(
            target_class,
            benchmark_spec,
            normalized_dataset,
            normalized_array_implementation,
            warmups=warmups,
            runs=runs,
        )
        candidate_samples = _collect_samples(
            target_class,
            benchmark_spec,
            normalized_dataset,
            normalized_array_implementation,
            warmups=warmups,
            runs=runs,
            kernel_override=submission_kernel,
        )

        baseline_median_ns = median(baseline_samples)
        candidate_median_ns = median(candidate_samples)

        result["correct"] = True
        result["status"] = "ok"
        result["baseline_samples_ms"] = [_ns_to_ms(sample) for sample in baseline_samples]
        result["candidate_samples_ms"] = [_ns_to_ms(sample) for sample in candidate_samples]
        result["baseline_median_ms"] = _ns_to_ms(baseline_median_ns)
        result["candidate_median_ms"] = _ns_to_ms(candidate_median_ns)
        result["speedup"] = baseline_median_ns / candidate_median_ns
        return result
    except InvalidSubmission as exc:
        result["status"] = "invalid"
        result["error"] = str(exc)
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _normalize_dataset_size(dataset_size: str) -> DataSetSize:
    try:
        return DataSetSize[str(dataset_size).upper()]
    except KeyError as exc:
        valid = ", ".join(size.name for size in DataSetSize)
        raise EvaluationError(f'Invalid dataset size "{dataset_size}". Expected one of: {valid}.') from exc


def _normalize_array_implementation(array_implementation: str) -> ArrayImplementation:
    normalized = str(array_implementation).strip().lower().replace("-", "_")
    mapping = {
        "numpy": ArrayImplementation.NUMPY,
        "list": ArrayImplementation.LIST,
    }
    if normalized not in mapping:
        valid = ", ".join(sorted(mapping))
        raise EvaluationError(
            f'Invalid array implementation "{array_implementation}". Expected one of: {valid}.'
        )
    return mapping[normalized]


def _validate_positive_int(name: str, value: int, allow_zero: bool) -> None:
    if not isinstance(value, int):
        raise EvaluationError(f'"{name}" must be an integer.')
    if allow_zero:
        if value < 0:
            raise EvaluationError(f'"{name}" must be >= 0.')
    elif value < 1:
        raise EvaluationError(f'"{name}" must be >= 1.')


def _canonical_benchmark_id(benchmark_id: str) -> str:
    normalized = str(benchmark_id).strip()
    if normalized.endswith(".py"):
        normalized = Path(normalized).stem
    if "." in normalized:
        normalized = normalized.split(".")[-1]
    normalized = normalized.replace("-", "_")
    if normalized and normalized[0].isdigit():
        normalized = f"_{normalized}"
    return normalized


def _resolve_benchmark(benchmark_id: str) -> tuple[type, PolyBenchSpec]:
    target_id = _canonical_benchmark_id(benchmark_id)
    spec_file = PolyBenchSpecFile("polybench.spec")
    specs_by_name = {_spec_to_python_name(spec): spec for spec in spec_file.specs}

    for benchmark_class in benchmark_classes:
        class_id = benchmark_class.__module__.split(".")[-1]
        if class_id == target_id:
            return benchmark_class, specs_by_name[class_id]

    raise UnknownBenchmark(f'Unknown benchmark "{benchmark_id}".')


def _spec_to_python_name(spec: PolyBenchSpec) -> str:
    name = spec.Name.replace("-", "_")
    if name and name[0].isdigit():
        name = f"_{name}"
    return name


def _instantiate_benchmark(
    benchmark_class: type,
    benchmark_spec: PolyBenchSpec,
    dataset_size: DataSetSize,
    array_implementation: ArrayImplementation,
):
    options = PolyBenchOptions()
    options.POLYBENCH_ARRAY_IMPLEMENTATION = array_implementation
    options.POLYBENCH_DATASET_SIZE = dataset_size
    return benchmark_class(options, benchmark_spec)


def _signature_shape(func) -> tuple[tuple[str, inspect._ParameterKind], ...]:
    return tuple((parameter.name, parameter.kind) for parameter in inspect.signature(func).parameters.values())


def _format_signature(func) -> str:
    return f"{func.__name__}{inspect.signature(func)}"


def _validate_submission_source(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise InvalidSubmission(f"Invalid Python submission: {exc.msg} (line {exc.lineno}).") from exc

    kernel_defs = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import_root(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                raise InvalidSubmission("Relative imports are not allowed in submissions.")
            _validate_import_root(node.module)
        elif isinstance(node, ast.AsyncFunctionDef) and node.name == "kernel":
            raise InvalidSubmission("Submission kernel must be a regular function, not async.")
        elif isinstance(node, ast.FunctionDef) and node.name == "kernel":
            kernel_defs.append(node)

    if len(kernel_defs) != 1:
        raise InvalidSubmission('Submission must define exactly one top-level function named "kernel".')


def _validate_import_root(module_name: str) -> None:
    root = module_name.split(".")[0]
    if root in _ALLOWED_NON_STDLIB_IMPORTS:
        return
    if root in sys.stdlib_module_names:
        return
    raise InvalidSubmission(f'Import "{module_name}" is not allowed. Only stdlib modules and NumPy are supported.')


def _load_submission_kernel(submission_source: str):
    _validate_submission_source(submission_source)

    temp_path = None
    module_name = f"polybench_submission_{uuid.uuid4().hex}"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="polybench_submission_",
            dir="/tmp",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(submission_source)
            temp_path = handle.name

        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        if spec is None or spec.loader is None:
            raise InvalidSubmission("Unable to create an import spec for the submission module.")

        module = importlib.util.module_from_spec(spec)
        module.__dict__.setdefault("np", np)
        module.__dict__.setdefault("numpy", np)
        module.__dict__.setdefault("ndarray", np.ndarray)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        kernel = getattr(module, "kernel", None)
        if not callable(kernel):
            raise InvalidSubmission('Submission module must define a callable top-level "kernel".')
        return kernel
    except InvalidSubmission:
        raise
    except Exception as exc:
        raise InvalidSubmission(f"Failed to load submission module: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _execute_once(
    benchmark_class: type,
    benchmark_spec: PolyBenchSpec,
    dataset_size: DataSetSize,
    array_implementation: ArrayImplementation,
    kernel_override=None,
) -> tuple[list[tuple[str, object]], int]:
    instance = _instantiate_benchmark(
        benchmark_class,
        benchmark_spec,
        dataset_size,
        array_implementation,
    )
    if kernel_override is not None:
        instance.kernel = types.MethodType(kernel_override, instance)

    elapsed_ns = None

    def _timed_kernel(self, *args, **kwargs):
        nonlocal elapsed_ns
        start = perf_counter_ns()
        self.kernel(*args, **kwargs)
        elapsed_ns = perf_counter_ns() - start

    instance.time_kernel = types.MethodType(_timed_kernel, instance)
    outputs = instance.run_benchmark()
    if elapsed_ns is None:
        raise EvaluationError("Benchmark did not execute the kernel through time_kernel().")
    return outputs, elapsed_ns


def _collect_samples(
    benchmark_class: type,
    benchmark_spec: PolyBenchSpec,
    dataset_size: DataSetSize,
    array_implementation: ArrayImplementation,
    warmups: int,
    runs: int,
    kernel_override=None,
) -> list[int]:
    for _ in range(warmups):
        _execute_once(
            benchmark_class,
            benchmark_spec,
            dataset_size,
            array_implementation,
            kernel_override=kernel_override,
        )

    samples = []
    for _ in range(runs):
        _, elapsed_ns = _execute_once(
            benchmark_class,
            benchmark_spec,
            dataset_size,
            array_implementation,
            kernel_override=kernel_override,
        )
        samples.append(elapsed_ns)
    return samples


def _compare_outputs(
    baseline_outputs: list[tuple[str, object]],
    candidate_outputs: list[tuple[str, object]],
    rtol: float,
    atol: float,
) -> tuple[bool, str | None]:
    if len(baseline_outputs) != len(candidate_outputs):
        return False, "Candidate returned a different number of outputs."

    for index, (baseline_output, candidate_output) in enumerate(zip(baseline_outputs, candidate_outputs), start=1):
        baseline_name, baseline_value = baseline_output
        candidate_name, candidate_value = candidate_output

        if baseline_name != candidate_name:
            return False, f'Output #{index} name mismatch: expected "{baseline_name}", received "{candidate_name}".'

        baseline_array = np.asarray(baseline_value)
        candidate_array = np.asarray(candidate_value)

        if baseline_array.shape != candidate_array.shape:
            return False, (
                f'Output "{baseline_name}" shape mismatch: expected {baseline_array.shape}, '
                f"received {candidate_array.shape}."
            )

        if np.issubdtype(baseline_array.dtype, np.number) and np.issubdtype(candidate_array.dtype, np.number):
            if not np.allclose(baseline_array, candidate_array, rtol=rtol, atol=atol, equal_nan=True):
                max_abs_diff = float(np.max(np.abs(baseline_array - candidate_array)))
                return False, (
                    f'Output "{baseline_name}" does not match the baseline within tolerance; '
                    f"max absolute difference {max_abs_diff}."
                )
        elif not np.array_equal(baseline_array, candidate_array):
            return False, f'Output "{baseline_name}" does not match the baseline.'

    return True, None


def _ns_to_ms(duration_ns: int | float) -> float:
    return float(duration_ns) / 1_000_000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an LLM-generated PolyBench kernel implementation.")
    parser.add_argument("--benchmark", required=True, help="Benchmark id such as gemm, jacobi_2d, or _2mm.")
    parser.add_argument(
        "--submission-file",
        help="Path to a Python source file containing the submission module. If omitted, stdin is used.",
    )
    parser.add_argument("--dataset-size", default="LARGE", help="Dataset size to evaluate.")
    parser.add_argument(
        "--array-implementation",
        default="numpy",
        choices=["numpy", "list"],
        help="Benchmark array implementation to evaluate against.",
    )
    parser.add_argument("--warmups", type=int, default=1, help="Number of untimed warmup runs.")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for correctness checks.")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for correctness checks.")
    parser.add_argument("--json-out", help="Optional path to write the JSON result.")
    args = parser.parse_args(argv)

    if args.submission_file:
        submission_source = load_submission_source(args.submission_file)
    else:
        submission_source = sys.stdin.read()

    result = evaluate_kernel(
        benchmark_id=args.benchmark,
        submission_source=submission_source,
        dataset_size=args.dataset_size,
        array_implementation=args.array_implementation,
        warmups=args.warmups,
        runs=args.runs,
        rtol=args.rtol,
        atol=args.atol,
    )

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
