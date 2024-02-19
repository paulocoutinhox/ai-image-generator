import platform

import psutil


def is_mac_arm():
    if platform.system() == "Darwin":
        if platform.machine() == "arm64":
            return True

    return False


def memory_less_64gb():
    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024**3)

    if total_ram_gb < 64:
        return True
    else:
        return False
