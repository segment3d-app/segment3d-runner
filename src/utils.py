import pynvml


def pick_available_gpus(count=2):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    usage_info = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        compute_util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        memory_usage = memory_info.used / memory_info.total / 100
        compute_usage = compute_util.gpu / 100

        weighted_score = (0.25 * memory_usage) + (0.75 * compute_usage)
        usage_info.append((i, weighted_score))

    usage_info.sort(key=lambda x: x[1])
    gpus = [str(gpu[0]) for gpu in usage_info[:count]]

    pynvml.nvmlShutdown()
    return gpus


def parse_command(command: str):
    return " ".join([line.strip() for line in command.split("\n") if line.strip()])
