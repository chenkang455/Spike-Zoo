from spikezoo.utils.spike_utils import load_vidar_dat
import time
print("cpp")
start = time.time()
for _ in range(10):
    result = load_vidar_dat("data/data.dat",version="cpp",width = 400,height = 250,out_format="array")
print(time.time() - start)

print("python")
start = time.time()
for _ in range(10):
    result = load_vidar_dat("data/data.dat",version="python",width = 400,height = 250,out_format="array")
print(time.time() - start)