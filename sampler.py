import time
import threading
import pandas as pd
import pynvml

class PowerSampler:
    """ polls gpu power, temp and clocks from nvml in a background thread """

    def __init__(self, gpu=0, interval=0.01):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.interval = interval
        self.rows = []
        self.last_power = 0.0 # latest reading, printed in the training log
        self.running = False
        # on A100 (GA100) GetPowerUsage is already the instant 25ms reading.
        # on H100 and newer it's a 1s average that erases microcycling, so use the instant field
        self.use_instant = 'A100' not in pynvml.nvmlDeviceGetName(self.handle)
        # an unsupported field doesn't raise, it just reads 0 W, so check once up front
        if self.use_instant and self.instant_field().nvmlReturn != pynvml.NVML_SUCCESS:
            print('warning: no instant power field on this gpu/driver, falling back to GetPowerUsage (may be a 1s average)')
            self.use_instant = False

    def instant_field(self):
        return pynvml.nvmlDeviceGetFieldValues(self.handle, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0]

    def read_power(self):
        if self.use_instant:
            return self.instant_field().value.uiVal / 1000 # mW -> W
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000 # mW -> W

    def energy_j(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000 # mJ -> J

    def loop(self):
        while self.running:
            t = time.perf_counter()
            self.last_power = self.read_power()
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
            self.rows.append((t, self.last_power, temp, sm_clock, throttle))
            time.sleep(max(0.0, self.interval - (time.perf_counter() - t)))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def save(self, path):
        pd.DataFrame(self.rows, columns=['t', 'power_w', 'temp_c', 'sm_clock_mhz', 'throttle']).to_csv(path, index=False)

def gpu_info(gpu=0):
    """ everything needed to report which physical gpu a run was on """
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    info = {
        'name': pynvml.nvmlDeviceGetName(h),
        'uuid': pynvml.nvmlDeviceGetUUID(h),
        'vbios': pynvml.nvmlDeviceGetVbiosVersion(h),
        'driver': pynvml.nvmlSystemGetDriverVersion(),
        'tdp_w': pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000,
        'power_limit_w': pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
    }
    try:
        info['serial'] = pynvml.nvmlDeviceGetSerial(h)
    except pynvml.NVMLError:
        info['serial'] = None # some cloud vms hide it, the uuid is still unique
    return info
