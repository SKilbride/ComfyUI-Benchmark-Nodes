import argparse
import subprocess
import time
import requests
import json
import uuid
import os
import sys
import yaml
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from threading import Thread, Event, Lock
from queue import Queue
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.workflow_manager import WorkflowManager
from core.package_manager import PackageManager
from core.gui import run_gui
from core.gui import show_restart_required_dialog
from core.vram_monitor import VRAMMonitor, PYNVML_AVAILABLE


class YamlObject:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.data = None
        self.load()

    def load(self):
        if os.path.exists(self.yaml_path):
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                self.data = yaml.safe_load(f) or {}
        else:
            self.data = {}

    def exists(self):
        return os.path.exists(self.yaml_path)

    def get(self, key, default=None):
        if self.data is None:
            self.load()
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key, value):
        if self.data is None:
            self.load()
        keys = key.split('.')
        d = self.data
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
            if not isinstance(d, dict):
                raise ValueError(f"Cannot set nested key '{key}' because intermediate '{k}' is not a dict")
        d[keys[-1]] = value
        self.save()

    def save(self):
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.data, f)

    def key_exists(self, key):
        if self.data is None:
            self.load()
        keys = key.split('.')
        value = self.data
        for k in keys:
            if not isinstance(value, dict):
                return False
            if k not in value:
                return False
            value = value[k]
        return True


class BenchmarkNodeManager:
    def __init__(self, custom_nodes_path):
        self.custom_nodes_path = custom_nodes_path
        self.benchmark_path = os.path.join(custom_nodes_path, 'comfyui-benchmark')
        self.exists = os.path.exists(self.benchmark_path) and os.path.isdir(self.benchmark_path)
        self.yaml = None
        if self.exists:
            yaml_path = os.path.join(self.benchmark_path, 'config.yaml')
            self.yaml = YamlObject(yaml_path)

def get_comfy_python(comfy_path: Path) -> str:
    """
    Return the correct python executable for a ComfyUI installation.
    Prioritizes: .venv > venv > conda (fallback to current)
    """
    comfy_path = Path(comfy_path)

    # 1. Standard venv (.venv or venv)
    candidates = [
        comfy_path / ".venv" / "Scripts" / "python.exe",
        comfy_path / ".venv" / "bin" / "python",
        comfy_path / "venv" / "Scripts" / "python.exe",
        comfy_path / "venv" / "bin" / "python",
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[INFO] Using ComfyUI's virtual environment: {cand}")
            return str(cand)

    # 2. Fallback: hope we're already in the right env
    print(f"[WARNING] No .venv found in {comfy_path} — using current Python ({sys.executable})")
    return sys.executable

def wait_for_completion(prompt_id, server_address, timeout=600, instance_id=None, debug=False):
    """
    Wait for prompt completion and extract execution time from History API timestamps.
    Returns: (history, exec_time_seconds or None)
    """
    start_time = time.time()
    last_health_check = 0
    health_check_interval = 15

    if debug:
        print(f"[DEBUG] [{instance_id}] Waiting for prompt {prompt_id} on {server_address} (timeout: {timeout}s)")

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time

        if debug and elapsed - last_health_check > health_check_interval:
            try:
                health = requests.get(f"http://{server_address}/system_stats", timeout=5)
                if health.status_code == 200:
                    stats = health.json()
                    vram_total = stats.get("vram_total", "N/A")
                    vram_free = stats.get("vram_free", "N/A")
                    print(f"[HEALTH] [{instance_id}] VRAM: {vram_total} MB, Free: {vram_free} MB")
                last_health_check = time.time()
            except Exception as e:
                print(f"[HEALTH] [{instance_id}] Failed: {e}")

        try:
            response = requests.get(f"http://{server_address}/history/{prompt_id}", timeout=8)
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    wall_clock_time = time.time() - start_time
                    prompt_data = history[prompt_id]
                    
                    # Extract execution time from status.messages timestamps
                    exec_time = None
                    status = prompt_data.get("status", {})
                    messages = status.get("messages", [])
                    
                    # Look for execution_start and execution_success timestamps
                    start_ts = None
                    end_ts = None
                    
                    for msg in messages:
                        if isinstance(msg, list) and len(msg) >= 2:
                            msg_type = msg[0]
                            msg_data = msg[1] if isinstance(msg[1], dict) else {}
                            
                            if msg_type == "execution_start" and "timestamp" in msg_data:
                                start_ts = msg_data["timestamp"]
                            elif msg_type == "execution_success" and "timestamp" in msg_data:
                                end_ts = msg_data["timestamp"]
                    
                    # Calculate execution time from timestamps (in milliseconds)
                    if start_ts is not None and end_ts is not None:
                        exec_time = (end_ts - start_ts) / 1000.0  # Convert ms to seconds
                    
                    if exec_time is not None:
                        print(f"Prompt {prompt_id} completed in {wall_clock_time:.2f}s (exec: {exec_time:.2f}s)")
                    else:
                        print(f"Prompt {prompt_id} completed in {wall_clock_time:.2f}s")
                    
                    return history, exec_time
                elif debug:
                    print(f"[DEBUG] [{instance_id}] Prompt in history but not complete")
            elif response.status_code == 404 and debug:
                print(f"[DEBUG] [{instance_id}] History not found yet")
            elif debug:
                print(f"[DEBUG] [{instance_id}] History status: {response.status_code}")
        except requests.Timeout:
            if debug:
                print(f"[TIMEOUT] [{instance_id}] Request timeout for /history/{prompt_id}")
        except requests.ConnectionError:
            if debug:
                print(f"[ERROR] [{instance_id}] Connection lost to {server_address}")
        except Exception as e:
            if debug:
                print(f"[ERROR] [{instance_id}] Wait error: {e}")

        time.sleep(2)

    raise TimeoutError(f"[{instance_id}] Prompt {prompt_id} timed out after {timeout}s on {server_address}")


def load_baseconfig(comfy_path, temp_dir=None, log_file=None):
    baseconfig_path = Path(comfy_path) / "baseconfig.json"
    paths_to_check = [baseconfig_path]
    if temp_dir:
        paths_to_check.append(Path(temp_dir) / "baseconfig.json")
    selected_path = None
    for path in paths_to_check:
        if path.exists():
            selected_path = path
            break
    if not selected_path:
        print(f"Warning: baseconfig.json not found. Using defaults.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Warning: baseconfig.json not found. Using defaults.\n")
        return {"NUM_INSTANCES": 1, "GENERATIONS": 1}
    try:
        with open(selected_path, 'r', encoding='utf-8') as f:
            baseconfig = json.load(f)
        if not isinstance(baseconfig, dict):
            raise ValueError("Invalid baseconfig.json")
        config_values = {
            "NUM_INSTANCES": baseconfig.get("NUM_INSTANCES", 1),
            "GENERATIONS": baseconfig.get("GENERATIONS", 1)
        }
        print(f"Loaded baseconfig: NUM_INSTANCES={config_values['NUM_INSTANCES']}, GENERATIONS={config_values['GENERATIONS']}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Loaded baseconfig from {selected_path}\n")
        return config_values
    except Exception as e:
        print(f"Failed to load baseconfig: {e}. Using defaults.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Failed to load baseconfig: {e}\n")
        return {"NUM_INSTANCES": 1, "GENERATIONS": 1}


def interrupt_process(port):
    server_address = f"127.0.0.1:{port}"
    try:
        response = requests.post(f"http://{server_address}/interrupt", timeout=5)
        if response.status_code == 200:
            print(f"Sent interrupt to {server_address}")
            return True
        else:
            print(f"Interrupt failed: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Interrupt error: {e}")
        return False


def check_server_ready(port, timeout=60, interval=2):
    server_address = f"127.0.0.1:{port}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{server_address}/", timeout=5)
            if response.status_code == 200:
                print(f"Server on port {port} is ready")
                return True
        except requests.ConnectionError:
            time.sleep(interval)
    print(f"Error: Server on port {port} not ready after {timeout}s")
    return False


def check_server_running(port):
    server_address = f"127.0.0.1:{port}"
    try:
        response = requests.get(f"http://{server_address}/", timeout=3)
        if response.status_code == 200:
            print(f"Server already running on port {port}")
            return True
    except requests.ConnectionError:
        return False
    return False


def capture_execution_times(proc, output_queue, capture_event, print_lock, log_file=None):
    pattern = re.compile(r"Prompt executed in (\d+\.\d+) seconds")
    progress_pattern = re.compile(r"\d+%\|.*?\| \d+/\d+ \[\d+:\d+.*?\d+\.\d+(?:it/s|s/it)\]")
    error_pattern = re.compile(r"!!! Exception during processing !!!")
    progress_buffer = []
    error_detected = False
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            try:
                line = line.decode('utf-8').strip()
            except:
                line = line.strip()
            line = line.replace('█', '#').replace('▌', '#').replace('▎', '#')
            
            # Check for execution time FIRST, before printing
            match = pattern.search(line)
            if match and capture_event.is_set() and not error_detected:
                exec_time = float(match.group(1))
                output_queue.put(exec_time)
            
            with print_lock:
                if error_pattern.search(line):
                    error_detected = True
                    print(line, flush=True)
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(line + '\n')
                elif progress_pattern.search(line):
                    progress_buffer.append(line)
                    print(f"\r{' ' * 120}\r{progress_buffer[-1]}", end='', flush=True)
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(progress_buffer[-1] + '\n')
                else:
                    if progress_buffer:
                        print(f"\n{progress_buffer[-1]}", flush=True)
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(progress_buffer[-1] + '\n')
                        progress_buffer = []
                    print(line, flush=True)
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(line + '\n')
    with print_lock:
        if progress_buffer:
            print(f"\n{progress_buffer[-1]}", flush=True)


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Run multiple ComfyUI instances concurrently.")
    parser.add_argument("-n", "--num_instances", type=int, default=1)
    parser.add_argument("-c", "--comfy_path", type=str, help="Path to ComfyUI folder")
    parser.add_argument("-w", "--workflow_path", type=str, help="Path to workflow ZIP or folder")
    parser.add_argument("-g", "--generations", type=int, default=1)
    parser.add_argument("-e", "--extract_minimal", action="store_true")
    parser.add_argument("-r", "--run_default", action="store_true")
    parser.add_argument("-o", "--override", type=str)
    parser.add_argument("-l", "--log", nargs='?', const=True, default=False)
    parser.add_argument("-t", "--temp_path", type=str)
    parser.add_argument("-p", "--port", type=int, default=8188)
    parser.add_argument("-u", "--use_main_workflow_only", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--debug-warmup", action="store_true", help="Enable verbose debug logging ONLY during warmup")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER)
    parser.add_argument("--force-extract", "-f", action="store_true", help="Force re-extraction of all models/nodes even if identical/newer")
    parser.add_argument("--gui", action="store_true", help="Show a Qt-based GUI to pick -c and -w")
    parser.add_argument("--timeout", type=int, default=4000, help="Timeout in seconds for prompt completion (default: 4000)")
    parser.add_argument("--vram-monitor", action="store_true", help="Enable automatic VRAM monitoring using pynvml")
    parser.add_argument("--vram-log", type=str, help="Path to save detailed VRAM CSV log")
    parser.add_argument("--vram-interval", type=float, default=0.5, help="VRAM sampling interval in seconds (default: 0.5)")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index to monitor (default: 0)")

    args = parser.parse_args()

    # === GUI MODE ===
    if args.gui:
        comfy_path_arg = Path(args.comfy_path).resolve() if args.comfy_path else None
        workflow_path_arg = Path(args.workflow_path).resolve() if args.workflow_path else None
        try:
            gui_result = run_gui(
                comfy_path=comfy_path_arg,
                workflow_path=workflow_path_arg,
                extract_minimal=args.extract_minimal,
                port=args.port,
                generations=args.generations,
                num_instances=args.num_instances,
                run_default=args.run_default,
                extra_args=args.extra_args or [],
                debug_warmup=args.debug_warmup,
                no_cleanup=args.no_cleanup,
                use_main_workflow_only=args.use_main_workflow_only,
                force_extract=args.force_extract,
                timeout=args.timeout,
                override=args.override
            )
        except SystemExit:
            sys.exit(0)
        # Apply GUI results
        args.comfy_path = str(gui_result['comfy_path'])
        args.workflow_path = str(gui_result['workflow_path'])
        args.extract_minimal = gui_result['extract_minimal']
        args.port = gui_result['port']
        args.generations = gui_result['generations']
        args.num_instances = gui_result['num_instances']
        args.run_default = gui_result['run_default']
        args.extra_args = gui_result['extra_args']
        args.debug_warmup = gui_result['debug_warmup']
        args.no_cleanup = gui_result['no_cleanup']
        args.use_main_workflow_only = gui_result['use_main_workflow_only']
        args.override = gui_result['override']
        args.timeout = gui_result['timeout']
        args.force_extract = gui_result['force_extract']
    else:
        if not args.comfy_path:
            parser.error("--comfy_path (-c) is required when not using --gui.")
        if not args.workflow_path:
            parser.error("--workflow_path (-w) is required when not using --gui.")

    # === LOG FILE ===
    log_file = None
    if args.log is not False:
        workflow_basename = Path(args.workflow_path).stem
        timestamp = datetime.now().strftime("%y%m%d_") + str(int(time.time()))
        log_file = Path(args.log).resolve() if args.log is not True else Path(f"{workflow_basename}_{timestamp}.txt").resolve()
        if log_file.is_dir():
            log_file = log_file / f"{workflow_basename}_{timestamp}.txt"
        print(f"Logging to: {log_file}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Run started: {datetime.now()}\n")

    workflow_path = Path(args.workflow_path).resolve()
    warmup_workflow_path = None
    comfy_path = Path(args.comfy_path).resolve()
    package_manager = None
    extra_args = args.extra_args or []

    # === INIT LISTS ===
    processes = []
    ports = []

    # === TIMESTAMP FOR ENTIRE RUN ===
    run_timestamp = datetime.now().strftime("%H%M%S")

    try:
        # Handle workflow
        if workflow_path.is_dir():
            package_manager = PackageManager(
                folder_path=workflow_path,
                comfy_path=comfy_path,
                extract_minimal=args.extract_minimal,
                force_extract=args.force_extract,
                log_file=log_file
            )
            workflow_path = package_manager.process_folder(workflow_path)
            warmup_workflow_path = workflow_path.parent / "warmup.json" if not args.use_main_workflow_only else workflow_path
            if not warmup_workflow_path.exists():
                warmup_workflow_path = workflow_path
        elif workflow_path.suffix.lower() == '.zip':
            package_manager = PackageManager(
                zip_path=workflow_path,
                comfy_path=comfy_path,
                temp_path=args.temp_path,
                extract_minimal=args.extract_minimal,
                force_extract=args.force_extract,
                log_file=log_file
            )
            workflow_path = package_manager.extract_zip()
            warmup_workflow_path = workflow_path.parent / "warmup.json" if not args.use_main_workflow_only else workflow_path
            if not warmup_workflow_path.exists():
                warmup_workflow_path = workflow_path
        elif workflow_path.suffix.lower() == '.json':
            # Handle standalone .json file
            warmup_workflow_path = workflow_path.parent / "warmup.json" if not args.use_main_workflow_only else workflow_path
            if not warmup_workflow_path.exists():
                warmup_workflow_path = workflow_path
        else:
            raise ValueError("Invalid workflow path: must be .zip, .json, or directory")

        workflow_manager = WorkflowManager(workflow_path=workflow_path, log_file=log_file)
        workflow_manager.load_workflow()
        warmup_workflow_manager = WorkflowManager(workflow_path=warmup_workflow_path, log_file=log_file)
        warmup_workflow_manager.load_workflow()

        if args.override:
            workflow_manager.apply_overrides(args.override)
            warmup_workflow_manager.apply_overrides(args.override)

        # === SAFETENSORS VALIDATION ===
        workflow_safetensors = workflow_manager.get_safetensors_files()
        if package_manager and package_manager.has_manifest and package_manager.temp_dir:
            for _mname in ("manifest.yaml", "manifest.yml", "manifest.json"):
                _mpath = package_manager.temp_dir / _mname
                if _mpath.exists():
                    workflow_manager.validate_manifest_safetensors(_mpath)
                    break

        benchmark_node_manager = BenchmarkNodeManager(custom_nodes_path=comfy_path / "custom_nodes")

        num_instances = args.num_instances
        generations = args.generations
        if args.run_default:
            bc = load_baseconfig(comfy_path, package_manager.temp_dir if package_manager else None, log_file)
            if args.num_instances == 1: num_instances = bc["NUM_INSTANCES"]
            if args.generations == 1: generations = bc["GENERATIONS"]

        base_port = args.port
        output_queues = [Queue() for _ in range(num_instances)]
        capture_events = [Event() for _ in range(num_instances)]
        print_lock = Lock()

        print(f"Starting {num_instances} instances...")
        for i in range(num_instances):
            port = base_port + i
            ports.append(port)
            if check_server_running(port):
                # FIX: Check the top-level package_manager flag, not just the nested extractor flag.
                # The PackageManager aggregates results from both the SmartExtractor AND the ManifestIntegration.
                if package_manager and getattr(package_manager, "custom_nodes_extracted", False):
                    print("Custom nodes were installed while a server was already running.  ComfyUI must be restarted before running the benchmark.")
                    show_restart_required_dialog(
                        package_manager=package_manager,
                        args=args,
                        python_exe=get_comfy_python(comfy_path),
                        script_fpath=(__file__),
                        log_file=log_file

                    )

                    # Optionally: exit after dialog so user can restart
                    print("Exiting. Please restart ComfyUI and re-run the command.")
                    sys.exit(0)
                processes.append(None)
                continue
            python_exe = get_comfy_python(comfy_path)
            cmd = [
                python_exe,
                "main.py",
                "--port", str(port),
                "--disable-xformers",
                *extra_args
            ]
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            proc = subprocess.Popen(
                cmd, cwd=comfy_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                text=True, encoding='utf-8', errors='replace', bufsize=1, env=env
            )
            processes.append(proc)
            print(f"Started instance {i+1} on port {port} (PID: {proc.pid})")
            Thread(target=capture_execution_times, args=(proc, output_queues[i], capture_events[i], print_lock, log_file), daemon=True).start()

        for i, port in enumerate(ports):
            if processes[i] is None: continue
            if not check_server_ready(port): raise RuntimeError(f"Server {port} failed")

        client_ids = [str(uuid.uuid4()) for _ in ports]

        def generation_task(idx, gen, is_warmup=False):
            instance_id = f"Inst{idx+1}"
            port = ports[idx]
            server_address = f"127.0.0.1:{port}"
            client_id = client_ids[idx]
            debug = args.debug_warmup and is_warmup

            prefix_type = "WARMUP" if is_warmup else "RUN"
            gen_num = 1 if is_warmup else gen + 1
            instance_num = idx + 1

            try:
                # DON'T enable stdout capture - we'll use History API instead
                # (stdout capture would create duplicates)
                
                print(f"[{instance_id}] Starting {'warmup' if is_warmup else f'gen {gen+1}'}")
                manager = warmup_workflow_manager if is_warmup else workflow_manager
                workflow = manager.get_workflow(randomize_seeds=True)

                # === UPDATE filename_prefix ===
                workflow = manager.update_filename_prefixes_in_copy(workflow, prefix_type, instance_num, gen_num, run_timestamp)

                if benchmark_node_manager.exists:
                    # Find existing Benchmark Workflow node
                    bid = next((nid for nid, n in workflow.items() if n.get("_meta", {}).get("title") == "Benchmark Workflow"), None)
                    
                    # Only modify if node already exists in workflow - don't create new ones
                    if bid:
                        node = workflow[bid]
                        node["inputs"]["file_postfix"] = "_warmup_" if is_warmup else f"_RUN_{gen+1}.{idx+1}"

                print(f"[{instance_id}] Queueing...")
                resp = requests.post(f"http://{server_address}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
                if resp.status_code != 200:
                    raise Exception(f"Queue failed: {resp.text}")
                prompt_id = resp.json()["prompt_id"]
                print(f"[{instance_id}] Queued: {prompt_id}")

                history, exec_time = wait_for_completion(prompt_id, server_address, timeout=args.timeout, instance_id=instance_id, debug=debug)

                # Store execution time for non-warmup runs (from History API only)
                if not is_warmup and exec_time:
                    output_queues[idx].put(exec_time)

                print(f"[{instance_id}] {'Warmup' if is_warmup else f'Gen {gen+1}'} completed")
            except Exception as e:
                print(f"[{instance_id}] ERROR: {e}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{instance_id}] ERROR: {e}\n")
                raise

        # === VRAM MONITOR SETUP ===
        vram_monitor = None
        vram_results = None
        if args.vram_monitor or args.vram_log:
            if PYNVML_AVAILABLE:
                # Generate CSV log path if not specified but monitoring is enabled
                csv_log_path = args.vram_log
                if not csv_log_path and args.vram_monitor:
                    workflow_basename = Path(args.workflow_path).stem
                    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
                    csv_log_path = str(Path(f"gpu_log_{workflow_basename}_{timestamp}.csv").resolve())
                
                vram_monitor = VRAMMonitor(
                    gpu_index=args.gpu_index,
                    sample_interval=args.vram_interval,
                    csv_log=csv_log_path,
                    log_file=log_file
                )
                # Capture idle VRAM before anything loads
                vram_monitor.capture_idle_vram()
                print(f"[VRAM Monitor] CSV log will be saved to: {csv_log_path}")
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"VRAM monitoring enabled, CSV log: {csv_log_path}\n")
            else:
                print("[WARNING] VRAM monitoring requested but pynvml not installed. Run: pip install pynvml")

        # === SAFE SEQUENTIAL WARMUP ===
        print("Performing SAFE sequential warmup to prevent VRAM OOM...")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("SAFE sequential warmup started\n")

        for idx in range(num_instances):
            port = ports[idx]
            instance_id = f"Inst{idx+1}"
            print(f"[WARMUP] Starting {instance_id} on port {port}")
            try:
                generation_task(idx, -1, is_warmup=True)
                print(f"[WARMUP] {instance_id} completed")
            except Exception as e:
                error_msg = f"[WARMUP] {instance_id} FAILED: {e}"
                print(error_msg)
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(error_msg + "\n")
                if "memory" in str(e).lower() or "cuda" in str(e).lower():
                    raise RuntimeError(f"CUDA OOM on {instance_id}. Reduce instances or use --lowvram.")
                raise
            finally:
                print(f"[WARMUP] Freeing VRAM on {instance_id}")
                interrupt_process(port)
                time.sleep(3)

        print("All warmups completed safely.")

        # Clear queues
        for q in output_queues:
            while not q.empty(): q.get()

        # Start VRAM monitoring for main benchmark
        if vram_monitor:
            vram_monitor.start()
            print("[VRAM Monitor] Started for main benchmark run")

        # MAIN RUN
        start_time = time.time()
        try:
            for gen in range(generations):
                print(f"Generation round {gen+1}/{generations}")
                with ThreadPoolExecutor(max_workers=num_instances) as executor:
                    futures = [executor.submit(generation_task, idx, gen, False) for idx in range(num_instances)]
                    for f in as_completed(futures):
                        f.result()
        except KeyboardInterrupt:
            print("Interrupted. Partial metrics...")
            raise

        # Stop VRAM monitoring and get results
        if vram_monitor:
            peak_vram, idle_vram, delta_vram = vram_monitor.stop()
            vram_results = vram_monitor.get_results_summary()
            print(f"[VRAM Monitor] Results: Peak={peak_vram:.0f}MB, Idle={idle_vram:.0f}MB, Delta={delta_vram:.0f}MB")

        # METRICS
        wall_clock_time = time.time() - start_time
        total_images = num_instances * generations
        exec_times = []
        for q in output_queues:
            while not q.empty():
                exec_times.append(q.get())
        
        if exec_times:
            total_exec = sum(exec_times)
            avg_exec = total_exec / len(exec_times)
            std_exec = statistics.stdev(exec_times) if len(exec_times) > 1 else 0.0
            apm = total_images / (total_exec / 60) if total_exec > 0 else 0
            atpi = total_exec / total_images if total_images > 0 else 0

            print(f"\nUsing History API timestamps: {len(exec_times)} measurements captured")
        else:
            # Fallback to wall clock time
            total_exec = wall_clock_time
            avg_exec = wall_clock_time / total_images if total_images > 0 else 0
            apm = total_images / (wall_clock_time / 60) if wall_clock_time > 0 else 0
            atpi = wall_clock_time / total_images if total_images > 0 else 0
            print(f"\nWarning: No execution times captured, using wall clock time (includes HTTP overhead)")

        # === PACKAGE NAME ===
        package_name = "N/A"
        if package_manager and hasattr(package_manager, 'package_name'):
            package_name = package_manager.package_name
        elif args.workflow_path:
            original_path = Path(args.workflow_path)
            package_name = original_path.stem
            if original_path.suffix.lower() == '.zip' and package_manager and package_manager.temp_dir:
                inner_dir = package_manager.temp_dir
                if inner_dir.exists():
                    package_name = inner_dir.name

        print("\n" + "#" * 50)
        print("####_RESULTS_SUMMARY_####")
        print(f"Benchmarking Package: {package_name}")
        if workflow_safetensors:
            print("Models:")
            for _sf in workflow_safetensors:
                print(f"  {_sf['filename']}  [{_sf['class_type']}]")
        if vram_results:
            print(f"VRAM: {vram_results['peak_vram_mb']:.0f} MB ({vram_results['peak_vram_gb']:.2f} GB) | Delta: {vram_results['delta_vram_mb']:.0f} MB")
        if num_instances > 1:
            print(f"Number of Concurrent ComfyUI Instances: {num_instances}")
            print(f"Number of Generations per Instance: {generations}")
        else:
            print(f"Number of Generations: {generations}")
        
        if exec_times:
            print(f"Workflow Execution Time: {total_exec:.2f}s | Benchmark Execution Time: {wall_clock_time:.2f}s")
            #print(f"Overhead: {wall_clock_time - total_exec:.2f}s ({((wall_clock_time - total_exec) / wall_clock_time * 100):.1f}%)")
            print(f"Assets Generated: {total_images} | Avg time per Asset: {atpi:.2f}s | Std Dev: {std_exec:.2f}s | APM: {apm:.2f}")
        else:
            print("****Could not access internal ComfyUI runtime data, benchmark results may not be accurate****")
            print(f"Total time: {wall_clock_time:.2f}s | Assets Generated: {total_images} | Avg/sec per Asset: {atpi:.2f} | APM: {apm:.2f}")
        print("#" * 50 + "\n")

        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('####_RESULTS_SUMMARY_####\n')
                f.write(f"Benchmarking Package: {package_name}\n")
                if workflow_safetensors:
                    f.write("Models:\n")
                    for _sf in workflow_safetensors:
                        f.write(f"  {_sf['filename']}  [{_sf['class_type']}]\n")
                if vram_results:
                    f.write(f"VRAM: {vram_results['peak_vram_mb']:.0f} MB ({vram_results['peak_vram_gb']:.2f} GB) | Delta: {vram_results['delta_vram_mb']:.0f} MB\n")
                if num_instances > 1:
                    f.write(f"Number of Concurrent ComfyUI Instances: {num_instances}\n")
                    f.write(f"Number of Image Generations per Instance: {generations}\n")
                else:
                    f.write(f"Number of Image Generations: {generations}\n")
                
                if exec_times:
                    f.write(f"Workflow Execution Time: {total_exec:.2f}s | Overall Benchmark Execution Time: {wall_clock_time:.2f}s\n")
                    #f.write(f"Overhead: {wall_clock_time - total_exec:.2f}s ({((wall_clock_time - total_exec) / wall_clock_time * 100):.1f}%)\n")
                    f.write(f"Assets Generated: {total_images} | Avg Execution time per Asset: {atpi:.2f}s | Std Dev: {std_exec:.2f}s | APM: {apm:.2f}\n")
                else:
                    f.write("****Could not access internal ComfyUI runtime data, benchmark results may not be accurate****")
                    f.write(f"Total time: {wall_clock_time:.2f}s | Assets Generated: {total_images} | Avg/sec per Asset: {atpi:.2f} | APM: {apm:.2f}\n")

    except KeyboardInterrupt:
        print("User interrupt.")
        # Stop VRAM monitor on interrupt
        if 'vram_monitor' in dir() and vram_monitor:
            try:
                vram_monitor.stop()
            except:
                pass
    finally:
        if args.no_cleanup:
            print("Skipping cleanup (--no-cleanup)")
        else:
            print("Cleaning up...")
            for i, proc in enumerate(processes):
                if not proc: continue
                port = base_port + i
                server = f"127.0.0.1:{port}"
                try:
                    q = requests.get(f"http://{server}/queue", timeout=3)
                    if q.status_code == 200 and (q.json().get("queue_running") or q.json().get("queue_pending")):
                        interrupt_process(port)
                        time.sleep(3)
                except: pass
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(10)
                    except: proc.kill()
            if package_manager: package_manager.cleanup()
            print("Cleanup done.")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Log: {log_file}\n")
        print("Done.")


if __name__ == "__main__":
    main()