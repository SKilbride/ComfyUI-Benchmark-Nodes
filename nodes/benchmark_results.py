import csv
import json
from datetime import datetime
from pathlib import Path

try:
    import folder_paths as _folder_paths
    _DEFAULT_OUTPUT_DIR = str(Path(_folder_paths.output_directory) / "benchmark_results")
except Exception:
    _DEFAULT_OUTPUT_DIR = "benchmark_results"


class BenchmarkResultsOutput:
    """
    Displays benchmark results in the ComfyUI server console and saves them
    to JSON (one file per run) and/or a rolling CSV (appended across runs).

    OUTPUT_NODE = True ensures this node always executes even if nothing
    connects to its outputs.
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "benchmark_results": ("BENCHMARK_RESULTS",),
            },
            "optional": {
                "output_dir": (
                    "STRING",
                    {"default": _DEFAULT_OUTPUT_DIR, "multiline": False},
                ),
                "save_json": ("BOOLEAN", {"default": True}),
                "save_csv": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "output_results"
    CATEGORY = "Benchmark"

    def output_results(
        self,
        benchmark_results,
        output_dir=_DEFAULT_OUTPUT_DIR,
        save_json=True,
        save_csv=True,
    ):
        config = benchmark_results.get("benchmark_config", {})
        test_group = config.get("test_group", "unknown")
        variant = config.get("variant", "unknown")
        success = benchmark_results.get("success", False)

        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  BENCHMARK RESULTS: {test_group} / {variant}")
        print(sep)
        if success:
            for key in ("resolution_line", "vram_line", "timing_line", "assets_line", "generations_line"):
                val = benchmark_results.get(key)
                if val:
                    print(f"  {val}")
        else:
            print(f"  FAILED (exit code {benchmark_results.get('returncode', '?')})")
        print(sep + "\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        safe_name = (
            f"{test_group}_{variant}".replace("/", "_").replace(" ", "_").replace("\\", "_")
        )

        if save_json:
            json_path = output_path / f"benchmark_{safe_name}_{timestamp}.json"
            # Exclude raw stdout to keep JSON file readable
            save_data = {k: v for k, v in benchmark_results.items() if k != "stdout"}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, default=str)
            print(f"[BenchmarkResults] JSON saved: {json_path}")

        if save_csv:
            csv_path = output_path / "benchmark_results.csv"
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "Timestamp", "Test Group", "Variant", "Success",
                        "Width", "Height", "Batch Size",
                        "Peak VRAM (MB)", "Peak VRAM (GB)", "Delta VRAM (MB)",
                        "Total Exec (s)", "Avg/Asset (s)", "Std Dev (s)",
                        "APM", "Total Assets",
                    ])
                writer.writerow([
                    timestamp,
                    test_group,
                    variant,
                    "Yes" if success else "No",
                    benchmark_results.get("width", ""),
                    benchmark_results.get("height", ""),
                    benchmark_results.get("batch_size", ""),
                    benchmark_results.get("peak_vram_mb", ""),
                    benchmark_results.get("peak_vram_gb", ""),
                    benchmark_results.get("delta_vram_mb", ""),
                    benchmark_results.get("total_exec_s", ""),
                    benchmark_results.get("avg_time_s", ""),
                    benchmark_results.get("std_dev_s", ""),
                    benchmark_results.get("apm", ""),
                    benchmark_results.get("total_assets", ""),
                ])
            print(f"[BenchmarkResults] CSV updated: {csv_path}")

        return ()
