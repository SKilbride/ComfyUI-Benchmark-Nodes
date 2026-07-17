import json
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"


def _get_test_groups():
    if not WORKFLOWS_DIR.exists():
        return ["(no workflows folder)"]
    groups = sorted(d.name for d in WORKFLOWS_DIR.iterdir() if d.is_dir())
    return groups if groups else ["(no test groups found)"]


def _get_all_variants():
    """
    Returns a placeholder variant list used only for server-side node registration.
    VALIDATE_INPUTS bypasses combo validation, and the frontend JS replaces this
    list with the real variants for the selected test group immediately on load.
    Hardcoding common names avoids showing confusing test-specific names
    (e.g. APPLE-MLX's 'FLUX1 Dev') as the initial default.
    """
    return ["BF16", "FP8", "FP16", "NVFP4", "GGUF_Q4", "GGUF_Q8"]


class BenchmarkSelector:
    """
    Selects a benchmark test group and variant from the node pack's workflows/ folder.

    The test_group combo is populated at server startup from the workflows/ directory.
    The variant combo is pre-populated with all variant names across all test groups
    so server-side validation accepts any value. The frontend JS widget narrows it
    to only the variants for the selected test group.
    """

    @classmethod
    def INPUT_TYPES(cls):
        test_groups = _get_test_groups()
        all_variants = _get_all_variants()
        return {
            "required": {
                "test_group": (test_groups, {"default": test_groups[0]}),
                "variant": (all_variants, {"default": all_variants[0]}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def VALIDATE_INPUTS(cls, test_group, variant):
        # Validation is handled at execution time in select(); accept any string here
        # so that saved workflows with any variant name reload without error.
        return True

    RETURN_TYPES = ("BENCHMARK_CONFIG",)
    RETURN_NAMES = ("benchmark_config",)
    FUNCTION = "select"
    CATEGORY = "Benchmark"
    OUTPUT_NODE = False

    def select(self, test_group: str, variant: str):
        testconfig_path = WORKFLOWS_DIR / test_group / "testconfig.json"
        if not testconfig_path.exists():
            raise FileNotFoundError(
                f"testconfig.json not found in workflows/{test_group}"
            )

        with open(testconfig_path, "r", encoding="utf-8") as f:
            testconfig = json.load(f)

        workflow_folder = None
        generations = 10
        width = -1
        height = -1
        batch_size = -1

        for test in testconfig.get("tests", []):
            for v in test.get("variants", []):
                if v.get("name") == variant:
                    workflow_folder = v["workflow"]
                    width = v.get("width", -1)
                    height = v.get("height", -1)
                    batch_size = v.get("batch_size", -1)
            cfg = test.get("config", {})
            generations = cfg.get("generations", generations)

        if workflow_folder is None:
            raise ValueError(
                f"Variant '{variant}' not found in test group '{test_group}'. "
                f"Available: {[v['name'] for t in testconfig.get('tests',[]) for v in t.get('variants',[])]}"
            )

        workflow_path = WORKFLOWS_DIR / test_group / workflow_folder
        if not workflow_path.exists():
            raise FileNotFoundError(
                f"Workflow folder not found: {workflow_path}"
            )

        config = {
            "test_group": test_group,
            "variant": variant,
            "workflow_folder": workflow_folder,
            "workflow_path": str(workflow_path),
            "generations": generations,
            "width": width,
            "height": height,
            "batch_size": batch_size,
        }
        return (config,)
