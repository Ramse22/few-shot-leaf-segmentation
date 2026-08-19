import sys, time
from datetime import datetime

# Run the normal inference with timing
sys.argv = ['GrowerInference.py', '../configs/config_inf_timing.yaml']

start = time.time()
exec(open('GrowerInference.py').read())
elapsed = time.time() - start

print(f"[TIMING][NEW INFERENCE] elapsed time: {elapsed:.3f} s")
with open("timing_results.txt", "a") as f:
    f.write(
        f"NEW_INFERENCE,{datetime.now().isoformat()},{elapsed:.4f},"
        f"n_locs=10000,batch_size=2048,window_size=64,device={device}\n"
    )
