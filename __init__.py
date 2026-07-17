import json
from pathlib import Path
from aiohttp import web

from .nodes.benchmark_selector import BenchmarkSelector
from .nodes.benchmark_runner import BenchmarkRunner
from .nodes.benchmark_results import BenchmarkResultsOutput
from .nodes.config_override import ConfigOverride

NODE_CLASS_MAPPINGS = {
    "BenchmarkSelector": BenchmarkSelector,
    "BenchmarkRunner": BenchmarkRunner,
    "BenchmarkResultsOutput": BenchmarkResultsOutput,
    "BenchmarkConfigOverride": ConfigOverride,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BenchmarkSelector": "Benchmark Selector",
    "BenchmarkRunner": "Benchmark Runner",
    "BenchmarkResultsOutput": "Benchmark Results Output",
    "BenchmarkConfigOverride": "Benchmark Config Override",
}

WEB_DIRECTORY = "web"

WORKFLOWS_DIR = Path(__file__).parent / "workflows"

# ---------------------------------------------------------------------------
# Server routes — provide test/variant data to the frontend JS widget
# ---------------------------------------------------------------------------
try:
    from server import PromptServer

    @PromptServer.instance.routes.get("/benchmark-nodes/tests")
    async def get_tests(request):
        """Return the list of available test group folder names."""
        if not WORKFLOWS_DIR.exists():
            return web.json_response({"tests": []})
        tests = sorted(d.name for d in WORKFLOWS_DIR.iterdir() if d.is_dir())
        return web.json_response({"tests": tests})

    @PromptServer.instance.routes.get("/benchmark-nodes/variants")
    async def get_variants(request):
        """
        Return variant names for a given test group.
        Query parameter: test=<test_group_folder_name>
        Response: {"variants": ["BF16", "FP8", ...]}
        """
        test_group = request.query.get("test", "").strip()
        if not test_group:
            return web.json_response(
                {"variants": [], "error": "Missing 'test' parameter"}, status=400
            )

        testconfig_path = WORKFLOWS_DIR / test_group / "testconfig.json"
        if not testconfig_path.exists():
            return web.json_response(
                {"variants": [], "error": "testconfig.json not found"}, status=404
            )

        try:
            with open(testconfig_path, "r", encoding="utf-8") as f:
                testconfig = json.load(f)
        except Exception as e:
            return web.json_response({"variants": [], "error": str(e)}, status=500)

        variants = [
            v["name"]
            for test in testconfig.get("tests", [])
            for v in test.get("variants", [])
            if v.get("name")
        ]
        return web.json_response({"variants": variants})

    print("[BenchmarkNodes] Server routes registered: /benchmark-nodes/tests, /benchmark-nodes/variants")

except Exception as e:
    print(f"[BenchmarkNodes] Warning: Could not register server routes: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
