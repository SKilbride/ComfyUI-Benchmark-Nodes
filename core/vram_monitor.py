"""
VRAM Monitor - Automated GPU memory monitoring for benchmarks.
Uses pynvml (NVIDIA Management Library) to capture VRAM usage.
"""

import time
import csv
import os
from threading import Thread, Event
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[WARNING] pynvml not installed. VRAM monitoring disabled. Install with: pip install pynvml")


class VRAMMonitor:
    """
    Background VRAM monitor that samples GPU memory usage at regular intervals.
    
    Usage:
        monitor = VRAMMonitor(gpu_index=0, sample_interval=0.5)
        monitor.start()
        # ... run benchmark ...
        peak_vram, idle_vram, delta_vram = monitor.stop()
        print(f"Peak VRAM: {peak_vram} MB, Delta: {delta_vram} MB")
    """
    
    def __init__(self, gpu_index: int = 0, sample_interval: float = 0.5, 
                 log_file: Optional[str] = None, csv_log: Optional[str] = None):
        """
        Initialize the VRAM monitor.
        
        Args:
            gpu_index: GPU device index (0 for first GPU)
            sample_interval: Time between samples in seconds (default 0.5s)
            log_file: Optional text log file path
            csv_log: Optional CSV log file path for detailed GPU-Z style logging
        """
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self.log_file = log_file
        self.csv_log = csv_log
        
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._samples: List[Tuple[datetime, float]] = []
        self._idle_vram: Optional[float] = None
        self._handle = None
        self._initialized = False
        
    def _init_nvml(self) -> bool:
        """Initialize NVML library."""
        if not PYNVML_AVAILABLE:
            return False
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._initialized = True
            
            # Get device name for logging
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            print(f"[VRAM Monitor] Initialized for GPU {self.gpu_index}: {name}")
            return True
        except Exception as e:
            print(f"[VRAM Monitor] Failed to initialize NVML: {e}")
            return False
    
    def _shutdown_nvml(self):
        """Shutdown NVML library."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                self._initialized = False
            except:
                pass
    
    def get_vram_used_mb(self) -> float:
        """Get current VRAM usage in MB."""
        if not self._initialized or not self._handle:
            return 0.0
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            print(f"[VRAM Monitor] Error reading VRAM: {e}")
            return 0.0
    
    def get_gpu_stats(self) -> dict:
        """Get comprehensive GPU stats (similar to GPU-Z)."""
        if not self._initialized or not self._handle:
            return {}
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            
            # Try to get temperature (may not be available on all GPUs)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0
            
            # Try to get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000  # mW to W
            except:
                power = 0
            
            # Try to get clock speeds
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM)
            except:
                gpu_clock = 0
                mem_clock = 0
                
            return {
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_total_mb': memory.total / (1024 * 1024),
                'memory_free_mb': memory.free / (1024 * 1024),
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory,
                'temperature_c': temp,
                'power_w': power,
                'gpu_clock_mhz': gpu_clock,
                'memory_clock_mhz': mem_clock,
            }
        except Exception as e:
            print(f"[VRAM Monitor] Error getting GPU stats: {e}")
            return {}
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        # Initialize CSV log if specified
        csv_file = None
        csv_writer = None
        if self.csv_log:
            try:
                csv_file = open(self.csv_log, 'w', newline='', encoding='utf-8')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    'Date', 'GPU Clock [MHz]', 'Memory Clock [MHz]', 
                    'GPU Temperature [C]', 'Memory Used [MB]', 'Memory Total [MB]',
                    'GPU Load [%]', 'Memory Controller Load [%]', 'Power Draw [W]'
                ])
            except Exception as e:
                print(f"[VRAM Monitor] Failed to create CSV log: {e}")
        
        while not self._stop_event.is_set():
            timestamp = datetime.now()
            stats = self.get_gpu_stats()
            
            if stats:
                vram_used = stats.get('memory_used_mb', 0)
                self._samples.append((timestamp, vram_used))
                
                # Write to CSV if enabled
                if csv_writer:
                    try:
                        csv_writer.writerow([
                            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            stats.get('gpu_clock_mhz', 0),
                            stats.get('memory_clock_mhz', 0),
                            stats.get('temperature_c', 0),
                            f"{vram_used:.1f}",
                            f"{stats.get('memory_total_mb', 0):.1f}",
                            stats.get('gpu_utilization', 0),
                            stats.get('memory_utilization', 0),
                            f"{stats.get('power_w', 0):.1f}"
                        ])
                        csv_file.flush()
                    except:
                        pass
            
            self._stop_event.wait(self.sample_interval)
        
        # Close CSV file
        if csv_file:
            try:
                csv_file.close()
            except:
                pass
    
    def capture_idle_vram(self, samples: int = 5, interval: float = 0.2) -> float:
        """
        Capture idle VRAM by taking multiple samples and averaging.
        Call this BEFORE starting the benchmark.
        """
        if not self._init_nvml():
            return 0.0
        
        readings = []
        for _ in range(samples):
            readings.append(self.get_vram_used_mb())
            time.sleep(interval)
        
        self._idle_vram = sum(readings) / len(readings)
        print(f"[VRAM Monitor] Idle VRAM: {self._idle_vram:.1f} MB")
        return self._idle_vram
    
    def start(self):
        """Start background VRAM monitoring."""
        if not self._initialized:
            if not self._init_nvml():
                return False
        
        # Capture idle VRAM if not already done
        if self._idle_vram is None:
            self.capture_idle_vram()
        
        self._stop_event.clear()
        self._samples = []
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[VRAM Monitor] Started monitoring (interval: {self.sample_interval}s)")
        return True
    
    def stop(self) -> Tuple[float, float, float]:
        """
        Stop monitoring and return results.
        
        Returns:
            Tuple of (peak_vram_mb, idle_vram_mb, delta_vram_mb)
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        
        # Calculate metrics
        if not self._samples:
            print("[VRAM Monitor] No samples collected")
            return (0.0, 0.0, 0.0)
        
        peak_vram = max(s[1] for s in self._samples)
        idle_vram = self._idle_vram or 0.0
        delta_vram = peak_vram - idle_vram
        
        print(f"[VRAM Monitor] Stopped. Collected {len(self._samples)} samples")
        print(f"[VRAM Monitor] Peak: {peak_vram:.1f} MB, Idle: {idle_vram:.1f} MB, Delta: {delta_vram:.1f} MB")
        
        self._shutdown_nvml()
        
        return (peak_vram, idle_vram, delta_vram)
    
    def get_samples(self) -> List[Tuple[datetime, float]]:
        """Get all collected samples."""
        return self._samples.copy()
    
    def get_peak_vram(self) -> float:
        """Get the peak VRAM usage from collected samples."""
        if not self._samples:
            return 0.0
        return max(s[1] for s in self._samples)
    
    def get_results_summary(self) -> dict:
        """Get a summary of the monitoring results."""
        if not self._samples:
            return {
                'peak_vram_mb': 0,
                'idle_vram_mb': 0,
                'delta_vram_mb': 0,
                'sample_count': 0,
                'peak_vram_gb': 0,
            }
        
        peak_vram = max(s[1] for s in self._samples)
        idle_vram = self._idle_vram or 0.0
        delta_vram = peak_vram - idle_vram
        
        return {
            'peak_vram_mb': round(peak_vram, 1),
            'idle_vram_mb': round(idle_vram, 1),
            'delta_vram_mb': round(delta_vram, 1),
            'sample_count': len(self._samples),
            'peak_vram_gb': round(peak_vram / 1024, 2),
        }


def format_vram_for_log(peak_mb: float, idle_mb: float = None) -> str:
    """Format VRAM stats for inclusion in benchmark log."""
    peak_gb = peak_mb / 1024
    if idle_mb is not None:
        delta_mb = peak_mb - idle_mb
        delta_gb = delta_mb / 1024
        return f"VRAM: {peak_mb:.0f} MB ({peak_gb:.2f} GB) | Peak Delta: {delta_mb:.0f} MB ({delta_gb:.2f} GB)"
    return f"VRAM: {peak_mb:.0f} MB ({peak_gb:.2f} GB)"


# Simple test
if __name__ == "__main__":
    print("Testing VRAM Monitor...")
    
    monitor = VRAMMonitor(gpu_index=0, sample_interval=0.5)
    
    # Capture idle VRAM
    idle = monitor.capture_idle_vram()
    print(f"Idle VRAM: {idle:.1f} MB")
    
    # Start monitoring
    monitor.start()
    
    # Simulate some work
    print("Monitoring for 5 seconds...")
    time.sleep(5)
    
    # Stop and get results
    peak, idle, delta = monitor.stop()
    
    print(f"\nResults:")
    print(f"  Peak VRAM: {peak:.1f} MB ({peak/1024:.2f} GB)")
    print(f"  Idle VRAM: {idle:.1f} MB")
    print(f"  Delta: {delta:.1f} MB ({delta/1024:.2f} GB)")
    
    print(f"\nFormatted: {format_vram_for_log(peak, idle)}")

