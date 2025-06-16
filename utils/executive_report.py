import asyncio
import json
import os
from datetime import datetime
from typing import Optional
from fastapi import BackgroundTasks
import psutil
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Interface for CPU metrics
class CpuMetrics:
    def __init__(
        self,
        manufacturer: str,
        brand: str,
        cores: int,
        speed: float,
        usage: float,
        temperature: Optional[float]
    ):
        self.manufacturer = manufacturer  # CPU manufacturer (e.g., Intel, AMD)
        self.brand = brand  # CPU brand/model (e.g., Core i7, Ryzen 5)
        self.cores = cores  # Number of physical CPU cores
        self.speed = speed  # CPU speed in GHz
        self.usage = usage  # Overall CPU usage percentage
        self.temperature = temperature  # CPU temperature in °C (None if unavailable)

# Interface for memory metrics
class MemoryCriticalMetrics:
    def __init__(
        self,
        total: float,
        used: float,
        free: float,
        swap_used: float,
        cache: float
    ):
        self.total = total  # Total physical memory in GB
        self.used = used  # Used memory in GB
        self.free = free  # Free memory in GB
        self.swap_used = swap_used  # Swap memory usage in GB
        self.cache = cache  # Cached memory in GB

# Interface for disk metrics
class DiskCriticalMetrics:
    def __init__(
        self,
        total: float,
        used: float,
        free: float,
        usage_percent: float
    ):
        self.total = total  # Total disk space in GB
        self.used = used  # Used disk space in GB
        self.free = free  # Free disk space in GB
        self.usage_percent = usage_percent  # Disk usage percentage

# Interface for network metrics
class NetworkCriticalMetrics:
    def __init__(
        self,
        rx_sec: float,
        tx_sec: float,
        errors: int,
        dropped: int
    ):
        self.rx_sec = rx_sec  # Network receive rate in MB/s
        self.tx_sec = tx_sec  # Network transmit rate in MB/s
        self.errors = errors  # Total network errors
        self.dropped = dropped  # Total dropped packets

# Interface for system metrics
class SystemCriticalMetrics:
    def __init__(
        self,
        uptime: float,
        os: str,
        kernel: str,
        hostname: str
    ):
        self.uptime = uptime  # System uptime in days
        self.os = os  # Operating system name and version
        self.kernel = kernel  # Kernel version
        self.hostname = hostname  # System hostname

async def get_cpu_metrics() -> CpuMetrics:
    # Note: CPU temperature requires platform-specific solutions
    # This is a simplified version that may not get manufacturer/brand on all systems
    cpu_freq = psutil.cpu_freq()
    return CpuMetrics(
        manufacturer="Unknown",  # Would need platform-specific code to get this
        brand="Unknown",  # Would need platform-specific code to get this
        cores=psutil.cpu_count(logical=False),
        speed=cpu_freq.current / 1000 if cpu_freq else 0,
        usage=psutil.cpu_percent(),
        temperature=None  # Would need platform-specific code to get this
    )

async def get_memory_metrics() -> MemoryCriticalMetrics:
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemoryCriticalMetrics(
        total=mem.total / (1024 ** 3),
        used=mem.used / (1024 ** 3),
        free=mem.available / (1024 ** 3),
        swap_used=swap.used / (1024 ** 3),
        cache=mem.cached / (1024 ** 3) if hasattr(mem, 'cached') else 0
    )

async def get_disk_metrics() -> DiskCriticalMetrics:
    disk = psutil.disk_usage('/')
    return DiskCriticalMetrics(
        total=disk.total / (1024 ** 3),
        used=disk.used / (1024 ** 3),
        free=disk.free / (1024 ** 3),
        usage_percent=disk.percent
    )

async def get_network_metrics() -> NetworkCriticalMetrics:
    net_io = psutil.net_io_counters()
    # Note: psutil doesn't provide per-second metrics directly, so this is total since boot
    # Would need to implement rate calculation by comparing with previous measurements
    return NetworkCriticalMetrics(
        rx_sec=0,  # Would need to implement rate calculation
        tx_sec=0,  # Would need to implement rate calculation
        errors=net_io.errin + net_io.errout,
        dropped=net_io.dropin + net_io.dropout
    )

async def get_system_metrics() -> SystemCriticalMetrics:
    try:
        # Try Unix-style first
        uname = os.uname()
        os_info = f"{uname.sysname} {uname.release}"
        kernel = uname.version
        hostname = uname.nodename
    except AttributeError:
        # Fallback for Windows
        import platform
        os_info = f"{platform.system()} {platform.release()}"
        kernel = platform.version()
        hostname = platform.node()
    
    return SystemCriticalMetrics(
        uptime=psutil.boot_time(),
        os=os_info,
        kernel=kernel,
        hostname=hostname
    )


async def get_executive_report():
    cpu_metrics = await get_cpu_metrics()
    mem_metrics = await get_memory_metrics()
    disk_metrics = await get_disk_metrics()
    system_metrics = await get_system_metrics()
    network_metrics = await get_network_metrics()

    return {
        "timestamp": datetime.now().isoformat(),
        "service": os.getenv("SERVICE_NAME", "Cartography Service"),
        "cpu_manufacturer": cpu_metrics.manufacturer,
        "cpu_brand": cpu_metrics.brand,
        "cpu_cores": cpu_metrics.cores,
        "cpu_speed": cpu_metrics.speed,
        "cpu_usage": cpu_metrics.usage,
        "cpu_temperature": cpu_metrics.temperature,
        "memory_total": mem_metrics.total,
        "memory_used": mem_metrics.used,
        "memory_free": mem_metrics.free,
        "memory_swap_used": mem_metrics.swap_used,
        "memory_cache": mem_metrics.cache,
        "disk_total": disk_metrics.total,
        "disk_used": disk_metrics.used,
        "disk_free": disk_metrics.free,
        "disk_usage_percent": disk_metrics.usage_percent,
        "network_rx_sec": network_metrics.rx_sec,
        "network_tx_sec": network_metrics.tx_sec,
        "network_errors": network_metrics.errors,
        "network_dropped": network_metrics.dropped,
        "system_uptime": system_metrics.uptime,
        "system_os": system_metrics.os,
        "system_kernel": system_metrics.kernel,
        "system_hostname": system_metrics.hostname
    }

async def write_report():
    try:
        report = await get_executive_report()
        
        # Get absolute path - more reliable than relative path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "../logs")
        path = os.path.join(log_dir, "executive_report.log")
        
        print(f"Attempting to write to: {path}")
        
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Write the file
        with open(path, "a") as f:
            f.write(json.dumps(report) + "\n")
        return {"status": "success", "message": f"Executive Report written to {path}"}
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"status": "error", "message": f"Failed to write executive report: {str(e)}"}

async def periodic_report():
        while True:
            try:
                await write_report()
            except Exception as e:
                print(f"ERROR in periodic report: {str(e)}")
            await asyncio.sleep(3600)
            
async def start_periodic_report_writer():
    asyncio.create_task(periodic_report())