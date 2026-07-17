import asyncio
import concurrent.futures
import json
import uuid
import time
import threading
import statistics
from pathlib import Path

try:
    from ..core.manifest_integration import integrate_manifest_from_folder
    _MANIFEST_INTEGRATION_AVAILABLE = True
except ImportError:
    _MANIFEST_INTEGRATION_AVAILABLE = False

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


def _run_async_in_thread(coro):
    """Run a coroutine in a fresh event loop in a background thread.

    ComfyUI 0.27+ runs node functions inside an async context where
    asyncio.run() raises 'cannot be called from a running event loop'.
    Spawning a dedicated thread with its own loop sidesteps this.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        future.result()  # blocks until done; propagates any exception


def _create_executor(execution_module, server):
    """Create a PromptExecutor compatible with the running ComfyUI version.

    ComfyUI 0.27 added cache_args to PromptExecutor.__init__ but left the
    default as None while execute_async unconditionally reads cache_args["ram"].
    We inspect the constructor and supply safe defaults when needed.
    """
    import inspect
    sig = inspect.signature(execution_module.PromptExecutor.__init__)
    kwargs = {}
    if "cache_args" in sig.parameters:
        # cache_type=False → not RAM_PRESSURE mode, but execute_async still
        # reads cache_args["ram"] unconditionally. 16.0 GB is a safe default.
        kwargs["cache_args"] = {"ram": 16.0, "ram_inactive": 0.0}
    return execution_module.PromptExecutor(server, **kwargs)


def _extract_execution_error(executor):
    """Pull the human-readable error message from executor.status_messages."""
    for event, data in getattr(executor, 'status_messages', []):
        if event == "execution_error":
            msg = data.get("exception_message", "")
            node_type = data.get("node_type", "")
            return f"[{node_type}] {msg}" if node_type else msg
    return "unknown error (check ComfyUI logs above)"


_LATENT_CREATOR_TYPES = {
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyFlux2LatentImage",
    "EmptyHunyuanLatentVideo",
    "EmptyLTXVLatentVideo",
    "EmptyLatentHunyuan3Dv2",
    "Wan22ImageToVideoLatent",
}


_SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced"}

# Node types that carry seed via a "noise_seed" input (e.g. FLUX / SamplerCustomAdvanced pipelines)
_NOISE_SEED_TYPES = {"RandomNoise", "DisableNoise"}


def _apply_seed(workflow, seed):
    """Update the seed in every sampler/noise node so each generation busts the executor cache.

    Covers KSampler-family (seed) and RandomNoise-family (noise_seed) so that
    both classic and FLUX-style workflows force re-execution while leaving model
    loader outputs cached from the first run.
    """
    for node in workflow.values():
        ct = node.get("class_type")
        if ct in _SAMPLER_TYPES:
            node["inputs"]["seed"] = seed
        elif ct in _NOISE_SEED_TYPES:
            node["inputs"]["noise_seed"] = seed


def _apply_latent_dimensions(workflow, width, height, batch_size):
    """Write effective width/height/batch_size into the workflow dict in-place.

    Each field is either an inline integer or a ["node_id", 0] reference to a
    PrimitiveInt node.  We update whichever form is present, and only for
    values that are not -1 (i.e. explicitly set).
    """
    for node in workflow.values():
        if node.get("class_type") not in _LATENT_CREATOR_TYPES:
            continue
        inputs = node.get("inputs", {})
        for field, value in (("width", width), ("height", height), ("batch_size", batch_size)):
            if field not in inputs or value == -1:
                continue
            current = inputs[field]
            if isinstance(current, list):
                ref_id = current[0]
                if ref_id in workflow:
                    workflow[ref_id]["inputs"]["value"] = value
            else:
                inputs[field] = value


def _vram_monitor_loop(handle, samples, stop_event, interval=0.1):
    while not stop_event.is_set():
        try:
            samples.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2))
        except Exception:
            pass
        time.sleep(interval)


class BenchmarkRunner:
    """
    Runs a benchmark workflow directly inside the current ComfyUI process.

    Uses ComfyUI's PromptExecutor directly — no subprocess, no port management,
    works with any ComfyUI installation including the Desktop Electron app.

    Accepts a BENCHMARK_CONFIG from BenchmarkSelector, executes the workflow
    N times, measures wall-clock time and VRAM, and returns structured results.

    This node blocks the ComfyUI prompt queue for the duration of the benchmark.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "benchmark_config": ("BENCHMARK_CONFIG",),
            },
            "optional": {
                "generations_override": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Override generation count (0 = use testconfig default)",
                    },
                ),
                "config_override": ("CONFIG_OVERRIDE",),
                "enable_vram_monitor": ("BOOLEAN", {"default": True}),
                "perform_warmup": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept any saved-workflow values without server-side type errors.
        # Type coercion happens in run_benchmark() itself.
        return True

    RETURN_TYPES = ("BENCHMARK_RESULTS",)
    RETURN_NAMES = ("benchmark_results",)
    FUNCTION = "run_benchmark"
    CATEGORY = "Benchmark"

    def run_benchmark(
        self,
        benchmark_config,
        generations_override=0,
        config_override=None,
        enable_vram_monitor=True,
        perform_warmup=True,
    ):
        from server import PromptServer
        import execution

        workflow_folder = Path(benchmark_config["workflow_path"])
        workflow_file = workflow_folder / "workflow.json"
        if not workflow_file.exists():
            raise FileNotFoundError(f"workflow.json not found: {workflow_file}")

        with open(workflow_file, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        try:
            generations_override = int(generations_override)
        except (TypeError, ValueError):
            generations_override = 0
        n_gens = generations_override if generations_override > 0 else benchmark_config["generations"]
        test_group = benchmark_config["test_group"]
        variant = benchmark_config["variant"]

        # Resolve effective width/height/batch_size: testconfig defaults → config_override
        eff_width = benchmark_config.get("width", -1)
        eff_height = benchmark_config.get("height", -1)
        eff_batch_size = benchmark_config.get("batch_size", -1)
        if config_override:
            if config_override.get("width", -1) != -1:
                eff_width = config_override["width"]
            if config_override.get("height", -1) != -1:
                eff_height = config_override["height"]
            if config_override.get("batch_size", -1) != -1:
                eff_batch_size = config_override["batch_size"]

        _apply_latent_dimensions(workflow, eff_width, eff_height, eff_batch_size)

        dim_str = (
            f"{eff_width}x{eff_height}" if eff_width != -1 and eff_height != -1
            else f"w={eff_width} h={eff_height}"
        )
        print(f"\n[BenchmarkRunner] {'='*55}")
        print(f"[BenchmarkRunner] Test:        {test_group} / {variant}")
        print(f"[BenchmarkRunner] Resolution:  {dim_str} | Batch: {eff_batch_size}")
        print(f"[BenchmarkRunner] Generations: {n_gens}")
        print(f"[BenchmarkRunner] Workflow:    {workflow_file}")
        print(f"[BenchmarkRunner] {'='*55}\n")

        # --- Manifest integration ---
        # Process manifest.yaml from the workflow folder (if present) to
        # download any missing models before execution starts.
        if _MANIFEST_INTEGRATION_AVAILABLE:
            try:
                import folder_paths
                comfy_path = Path(folder_paths.base_path)
                has_manifest, manifest_config, custom_nodes_installed = integrate_manifest_from_folder(
                    folder_path=workflow_folder,
                    comfy_path=comfy_path,
                )
                if has_manifest:
                    print(f"[BenchmarkRunner] Manifest processed for: {workflow_folder.name}")
                    if custom_nodes_installed:
                        print("[BenchmarkRunner] Custom nodes were installed — a ComfyUI restart may be required.")
                else:
                    print(f"[BenchmarkRunner] No manifest.yaml found in: {workflow_folder.name}")
            except Exception as exc:
                print(f"[BenchmarkRunner] WARNING: Manifest processing failed (benchmark will continue): {exc}")

        # VRAM monitoring
        vram_samples = []
        baseline_vram_mb = 0.0
        stop_vram = threading.Event()
        vram_thread = None

        if enable_vram_monitor and _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                baseline_vram_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
                vram_thread = threading.Thread(
                    target=_vram_monitor_loop,
                    args=(handle, vram_samples, stop_vram),
                    daemon=True,
                )
                vram_thread.start()
            except Exception as exc:
                print(f"[BenchmarkRunner] VRAM monitor unavailable: {exc}")

        # Find output node IDs (nodes with OUTPUT_NODE=True) so execute_async
        # knows which nodes to run. An empty list causes the execution loop to
        # exit immediately without running anything.
        import nodes as comfy_nodes
        output_node_ids = [
            nid for nid, ndata in workflow.items()
            if getattr(
                comfy_nodes.NODE_CLASS_MAPPINGS.get(ndata.get("class_type", "")),
                "OUTPUT_NODE",
                False,
            )
        ]
        if not output_node_ids:
            raise ValueError(
                "No OUTPUT_NODE found in workflow — add a SaveImage or similar output node."
            )
        print(f"[BenchmarkRunner] Output nodes: {output_node_ids}")

        # Run benchmark using ComfyUI's execution engine directly (no subprocess).
        # A single executor is reused across all generations so model loader outputs
        # (UNETLoader, CLIPLoader, VAELoader) stay in the node cache and are not
        # re-parsed from disk on every run. The seed is varied each generation so
        # the sampler and all downstream nodes re-execute despite the cached inputs
        # on unchanged upstream nodes.
        server = PromptServer.instance
        executor = _create_executor(execution, server)
        _gen_seed = [int(time.time()) & 0xFFFFFFFF]

        def _run_one(label):
            _apply_seed(workflow, _gen_seed[0])
            _gen_seed[0] = (_gen_seed[0] + 1) & 0xFFFFFFFF
            prompt_id = str(uuid.uuid4())
            t0 = time.perf_counter()
            if hasattr(executor, 'execute_async'):
                _run_async_in_thread(
                    executor.execute_async(workflow, prompt_id, {}, output_node_ids)
                )
            else:
                executor.execute(workflow, prompt_id, {}, output_node_ids)
            elapsed = time.perf_counter() - t0
            if not getattr(executor, 'success', True):
                err_msg = _extract_execution_error(executor)
                raise RuntimeError(f"Workflow execution failed on {label}: {err_msg}")
            return elapsed

        effective_batch = max(eff_batch_size, 1)  # treat -1 (unknown) as 1
        times = []
        error = None
        try:
            if perform_warmup:
                print("[BenchmarkRunner] Running warmup generation (discarded)...")
                warmup_elapsed = _run_one("warmup")
                print(f"[BenchmarkRunner] Warmup done in {warmup_elapsed:.3f}s")

            for i in range(n_gens):
                elapsed = _run_one(f"generation {i+1}/{n_gens}")
                times.append(elapsed)
                assets_so_far = (i + 1) * effective_batch
                print(f"[BenchmarkRunner] [{i+1}/{n_gens}] {elapsed:.3f}s ({assets_so_far} assets total)")
        except Exception as exc:
            error = exc
            print(f"[BenchmarkRunner] ERROR during execution: {exc}")
        finally:
            stop_vram.set()
            if vram_thread:
                vram_thread.join(timeout=2.0)

        if error and not times:
            raise error

        # Calculate stats — all per-image metrics scale by batch_size
        total = sum(times)
        total_assets = len(times) * effective_batch
        avg = total / total_assets if total_assets > 0 else 0.0
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        apm = (total_assets / total * 60.0) if total > 0 else 0.0
        peak_vram_mb = max(vram_samples) if vram_samples else 0.0
        delta_vram_mb = peak_vram_mb - baseline_vram_mb if vram_samples else 0.0

        vram_line = (
            f"Peak VRAM: {peak_vram_mb:.0f} MB ({peak_vram_mb/1024:.2f} GB)"
            f" | Delta: {delta_vram_mb:+.0f} MB"
            if vram_samples else "VRAM: not monitored"
        )
        timing_line = f"Total: {total:.1f}s | Avg/Asset: {avg:.3f}s | Std Dev: {std:.3f}s"
        assets_line = f"APM: {apm:.2f} | Total Assets: {total_assets}"
        generations_line = f"Completed {len(times)} of {n_gens} generations ({effective_batch} asset/gen)"
        resolution_line = f"Resolution: {dim_str} | Batch: {eff_batch_size}"

        print(f"\n[BenchmarkRunner] {'='*55}")
        print(f"[BenchmarkRunner] {resolution_line}")
        print(f"[BenchmarkRunner] {vram_line}")
        print(f"[BenchmarkRunner] {timing_line}")
        print(f"[BenchmarkRunner] {assets_line}")
        print(f"[BenchmarkRunner] {'='*55}\n")

        results = {
            "benchmark_config": benchmark_config,
            "success": error is None and len(times) == n_gens,
            "returncode": 0 if error is None else 1,
            "stdout": "",
            "width": eff_width,
            "height": eff_height,
            "batch_size": eff_batch_size,
            "total_exec_s": round(total, 3),
            "avg_time_s": round(avg, 3),
            "std_dev_s": round(std, 3),
            "apm": round(apm, 2),
            "total_assets": total_assets,
            "peak_vram_mb": round(peak_vram_mb, 1),
            "peak_vram_gb": round(peak_vram_mb / 1024.0, 3),
            "delta_vram_mb": round(delta_vram_mb, 1),
            "resolution_line": resolution_line,
            "vram_line": vram_line,
            "timing_line": timing_line,
            "assets_line": assets_line,
            "generations_line": generations_line,
        }

        return (results,)
