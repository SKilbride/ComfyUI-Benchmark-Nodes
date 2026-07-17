import re
from typing import Dict


def parse_results_summary(stdout: str) -> Dict:
    """
    Extract structured data from the ####_RESULTS_SUMMARY_#### sentinel block
    written by run_comfyui_benchmark_framework.py.
    Returns an empty dict if the sentinel is not present.
    """
    if "####_RESULTS_SUMMARY_####" not in stdout:
        return {}

    section = stdout.split("####_RESULTS_SUMMARY_####", 1)[1]
    results: Dict = {}

    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if "Benchmarking Package:" in line:
            results["package"] = line.split(":", 1)[1].strip()

        elif "VRAM:" in line:
            results["vram_line"] = line
            m = re.search(r"VRAM:\s*(\d+)\s*MB\s*\(([\d.]+)\s*GB\)", line)
            if m:
                results["peak_vram_mb"] = int(m.group(1))
                results["peak_vram_gb"] = float(m.group(2))
            dm = re.search(r"Delta:\s*(\d+)\s*MB", line)
            if dm:
                results["delta_vram_mb"] = int(dm.group(1))

        elif "Assets Generated:" in line:
            results["assets_line"] = line
            m = re.search(r"Assets Generated:\s*(\d+)", line)
            if m:
                results["total_assets"] = int(m.group(1))
            for pattern, key in [
                (r"Avg time per Asset:\s*([\d.]+)s", "avg_time_s"),
                (r"Std Dev:\s*([\d.]+)s", "std_dev_s"),
                (r"APM:\s*([\d.]+)", "apm"),
            ]:
                m = re.search(pattern, line)
                if m:
                    results[key] = float(m.group(1))

        elif "Workflow Execution Time:" in line:
            results["timing_line"] = line
            m = re.search(r"Workflow Execution Time:\s*([\d.]+)s", line)
            if m:
                results["total_exec_s"] = float(m.group(1))

        elif "Number of Generations" in line:
            results["generations_line"] = line

    return results
