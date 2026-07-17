# ComfyUI Benchmark Nodes

A ComfyUI custom node pack for benchmarking image and video generation models. Runs benchmark workflows directly inside the active ComfyUI process — no separate ComfyUI instance, no subprocess, compatible with ComfyUI Desktop.

---

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes` directory:
   ```
   ComfyUI/custom_nodes/ComfyUI-Benchmark-Nodes/
   ```
2. Install Python dependencies using **ComfyUI's own Python environment**:

   **ComfyUI Desktop (Windows):**
   ```
   C:\Users\<username>\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\python_embeds\python.exe -m pip install -r requirements.txt
   ```

   **Standard ComfyUI install:**
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI.

The nodes will appear under the **Benchmark** category in the node browser.

> **VRAM monitoring** requires `pynvml`. It is included in `requirements.txt`. If VRAM readings show `0.0` after installation, the package may not be installed in the correct Python environment — use the same pip command as above and confirm it targets ComfyUI's Python, not a system Python.

---

## Workflow

The three required nodes connect left to right. The **Benchmark Config Override** node is optional.

```
[Benchmark Selector] ──► [Benchmark Runner] ──► [Benchmark Results Output]
                                ▲
                    [Benchmark Config Override]  (optional)
```

---

## Nodes

### Benchmark Selector

Selects which benchmark test group and variant to run. Outputs a `BENCHMARK_CONFIG` object passed to the Benchmark Runner.

| Widget | Description |
|---|---|
| `test_group` | The model family to benchmark (e.g. `FLUX2.KLEIN_4B-T2I-BASE`). Populated from the `workflows/` folder at startup. |
| `variant` | The precision/quantization variant to run (e.g. `FP8`, `NVFP4`, `BF16`). The dropdown updates automatically when `test_group` changes. |

The selector reads `testconfig.json` from the selected test group's folder to find the workflow path, default generation count, and default resolution for that variant.

> **Definition:** One *generation* = one full execution of the workflow, producing `batch_size` images (or video frames for T2V models). Total images = generations × batch_size.

---

### Benchmark Config Override *(optional)*

Overrides the default width, height, or batch size defined in `testconfig.json` for the selected variant. Connect its `config_override` output to the `config_override` input on Benchmark Runner.

Leave any field at **-1** to use the testconfig default for that dimension.

| Widget | Default | Description |
|---|---|---|
| `width` | `-1` | Output image width in pixels. -1 = use testconfig default. |
| `height` | `-1` | Output image height in pixels. -1 = use testconfig default. |
| `batch_size` | `-1` | Number of images per generation. -1 = use testconfig default. |

> **Note:** This node is optional. If not connected, the runner uses the width, height, and batch size from `testconfig.json`.

---

### Benchmark Runner

Executes the selected benchmark workflow N times inside the active ComfyUI process, measures wall-clock time and peak VRAM usage, and outputs a `BENCHMARK_RESULTS` object.

| Widget | Default | Description |
|---|---|---|
| `generations_override` | `0` | Number of generations to run. `0` = use the count from `testconfig.json`. |
| `config_override` | *(input)* | Optional connection from a Benchmark Config Override node. |
| `enable_vram_monitor` | `true` | Sample GPU VRAM usage throughout the run (requires `pynvml`). |
| `perform_warmup` | `true` | Run one extra generation before timing begins. The warmup result is discarded. Recommended to ensure models are fully loaded and caches are warm before measurements start. |

**What it does at runtime:**
1. Loads the workflow JSON for the selected variant.
2. Processes `manifest.yaml` (if present) to download any missing model files. **The first run of a new variant may download several GB of model weights** — allow extra time and ensure sufficient disk space before starting.
3. Applies the effective width, height, and batch size to the workflow's latent node.
4. Optionally runs a warmup generation (discarded from results).
5. Runs N timed generations using ComfyUI's execution engine directly.
6. Returns timing and VRAM statistics.

**Console output during a run:**
```
[BenchmarkRunner] =======================================================
[BenchmarkRunner] Test:        FLUX2.KLEIN_4B-T2I-BASE / NVFP4
[BenchmarkRunner] Resolution:  1024x1024 | Batch: 1
[BenchmarkRunner] Generations: 10
[BenchmarkRunner] Workflow:    ...\flux.2_klein_4b_base_NVFP4_1024x1024x20
[BenchmarkRunner] =======================================================

[BenchmarkRunner] Running warmup generation (discarded)...
[BenchmarkRunner] Warmup done in 29.341s
[BenchmarkRunner] [1/10] 14.823s (2 assets total)
[BenchmarkRunner] [2/10] 14.701s (4 assets total)
...

[BenchmarkRunner] =======================================================
[BenchmarkRunner] Resolution: 1024x1024 | Batch: 1
[BenchmarkRunner] VRAM: not monitored
[BenchmarkRunner] Total: 148.2s | Avg/Asset: 14.820s | Std Dev: 0.091s
[BenchmarkRunner] APM: 4.05 | Total Assets: 10
[BenchmarkRunner] =======================================================
```

> **Important:** The Benchmark Runner blocks the ComfyUI prompt queue for the entire duration of the benchmark. Do not queue other workflows while a benchmark is running.

---

### Benchmark Results Output

Prints a formatted summary to the ComfyUI server console and saves results to disk. This node is an **output node** — it always executes even if nothing is connected downstream.

| Widget | Default | Description |
|---|---|---|
| `output_dir` | `<ComfyUI output>/benchmark_results/` | Folder where JSON and CSV files are written. Defaults to an absolute path inside ComfyUI's output directory. |
| `save_json` | `true` | Save a timestamped JSON file for each run containing all result fields. |
| `save_csv` | `true` | Append a row to a rolling `benchmark_results.csv` file in `output_dir`. |

**Console output:**
```
==============================================================
  BENCHMARK RESULTS: FLUX2.KLEIN_4B-T2I-BASE / NVFP4
==============================================================
  Resolution: 1024x1024 | Batch: 1
  VRAM: not monitored
  Total: 148.2s | Avg/Asset: 14.820s | Std Dev: 0.091s
  APM: 4.05 | Total Assets: 10
  Completed 10 of 10 generations (1 asset/gen)
==============================================================
```

**JSON output** (one file per run, e.g. `benchmark_FLUX2.KLEIN_4B-T2I-BASE_NVFP4_260712_112000.json`):
```json
{
  "benchmark_config": { "test_group": "...", "variant": "...", "generations": 10, "width": 1024, "height": 1024, "batch_size": 1 },
  "success": true,
  "width": 1024,
  "height": 1024,
  "batch_size": 1,
  "total_exec_s": 148.2,
  "avg_time_s": 14.82,
  "std_dev_s": 0.091,
  "apm": 4.05,
  "total_assets": 10,
  "peak_vram_mb": 12345.0,
  "peak_vram_gb": 12.056,
  "delta_vram_mb": 8192.0,
  "resolution_line": "Resolution: 1024x1024 | Batch: 1",
  "timing_line": "Total: 148.2s | Avg/Asset: 14.820s | Std Dev: 0.091s",
  "assets_line": "APM: 4.05 | Total Assets: 10",
  "generations_line": "Completed 10 of 10 generations (1 asset/gen)"
}
```

**CSV output** (`benchmark_results.csv`) — one row per run:

| Column | Description |
|---|---|
| Timestamp | `YYMMDD_HHMMSS` |
| Test Group | e.g. `FLUX2.KLEIN_4B-T2I-BASE` |
| Variant | e.g. `NVFP4` |
| Success | `Yes` / `No` |
| Width / Height / Batch Size | Effective dimensions used |
| Peak VRAM (MB) / (GB) | Highest GPU memory observed during the run |
| Delta VRAM (MB) | Peak minus baseline (memory consumed by the model load) |
| Total Exec (s) | Wall-clock time for all timed generations |
| Avg/Asset (s) | Average seconds per asset (`total / total_assets`) |
| Std Dev (s) | Standard deviation of per-generation times |
| APM | Assets per minute (`total_assets / total_exec × 60`). An asset is one image, video, or other generated output. |
| Total Assets | `generations × batch_size` |

> **CSV header:** The header row is only written when the file is first created. If you have an existing `benchmark_results.csv` with old column names, delete it and the correct header will be written on the next run.

---

## Available Test Groups

| Test Group | Models |
|---|---|
| `FLUX2.KLEIN_4B-T2I-BASE` | FLUX.2 Klein 4B Base — FP16, FP8, NVFP4, GGUF Q8 |
| `LTX2.3-T2V` | LTX Video 2.3 — BF16, FP8, NVFP4 |


## Model Downloads

Each workflow variant includes a `manifest.yaml` that lists required model files. When a benchmark runs, the runner automatically checks for missing models and downloads them from HuggingFace before execution begins. For details on how to author a `manifest.yaml` for your own test cases, see [create_testcase.md](create_testcase.md).

For models behind a HuggingFace access gate, set the `HF_TOKEN` environment variable before starting ComfyUI:

```bash
# Windows
set HF_TOKEN=hf_your_token_here

# Linux / macOS
export HF_TOKEN=hf_your_token_here
```

---

## Results Location

By default, results are saved to:
```
<ComfyUI output directory>/benchmark_results/
```

On a standard ComfyUI Desktop install on Windows this is typically:
```
C:\Users\<username>\Documents\ComfyUI\output\benchmark_results\
```

The path is shown in the `output_dir` widget on the Benchmark Results Output node and can be changed to any absolute path.

---

#
- **ComfyUI Desktop compatible.** The runner uses ComfyUI's internal execution engine directly — it does not launch a subprocess or require access to `main.py`.
- **Queue blocking.** A benchmark run holds the ComfyUI queue for its full duration. Other queued prompts will not run until it completes.
- **Fresh executor per generation.** A new `PromptExecutor` is created for each generation so the node cache never skips re-execution. Model weights stay in VRAM between generations via ComfyUI's global model management.
- **Warmup.** With `perform_warmup` enabled (default), the first generation is run but its time is not recorded. This ensures model weights and any JIT compilation are fully settled before timing starts.
- **Std Dev.** The standard deviation metric measures consistency between generations, not between individual images in a batch.
