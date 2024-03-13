import os
import time


def process(line):
    return line.strip().split(',')

# os.system('CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 6 gpu.py')
def query():
    cmd = 'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader'
    results = os.popen(cmd).readlines()
    to_numberic = lambda v: int(v.upper().strip().replace('%',''))
    results = [process(line) for line in results]
    for i in range(len(results)):
        results[i][1] = to_numberic(results[i][1])
    return sorted(results,key=lambda d: d[1], reverse=False)

flag = 0
while True:
    print("Querying...")
    result = query()
    i = 0
    gpus = []
    while i < len(result) and result[i][1] < 5:
        gpus.append(result[i][0])
        i += 1
    if len(gpus) > 0:
        if flag >= 2:
            flag = 0
        else:
            flag += 1
            time.sleep(120)
            continue
        port = 30000 + len(gpus)
        devices = ','.join(gpus)
        cmd = f'CUDA_VISIBLE_DEVICES={devices} python -m torch.distributed.launch --nproc_per_node {len(gpus)} --master_port {port} gpu.py &'
        # print(cmd)
        try:
            print(f"get gpu: {devices}")
            os.system(cmd)
        except:
            pass
    else:
        flag = 0
    time.sleep(120)
