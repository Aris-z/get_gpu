import os
import time
import setproctitle
setproctitle.setproctitle("python contrastive.py")


def process(line):
    return line.strip().split(',')

# os.system('CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 6 gpu.py')
def query():
    cmd = 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
    results = os.popen(cmd).readlines()
    to_numberic_1 = lambda v: int(v.upper().strip().replace('%',''))
    to_numberic_2 = lambda v: int(v.upper().strip().replace('MIB',''))
    results = [process(line) for line in results]
    for i in range(len(results)):
        results[i][1] = to_numberic_1(results[i][1])
        results[i][2] = to_numberic_2(results[i][2])
    return sorted(results,key=lambda d: d[1], reverse=False)

flag = 0
while True:
    result = query()
    i = 0
    gpus = []
    for i in range(len(result)):
        if result[i][1] < 10 and result[i][2] < 1000: #* GPU less than 10% or less than 1G*/
            gpus.append(result[i][0])
    print(f"Querying...GPUs' states are: {gpus}")
    if len(gpus) > 0:
        if flag >= 2:
            flag = 0
        else:
            flag += 1
            time.sleep(120)
            continue
        port = 30000 + len(gpus)
        devices = ','.join(gpus)
        cmd = f'CUDA_VISIBLE_DEVICES={devices} python -m torch.distributed.launch --nproc_per_node {len(gpus)} --master_port {port} /mnt/nas/data/yihan/get_gpu/train.py &'
        # print(cmd)
        try:
            print(f"get gpus: {devices}")
            os.system(cmd)
        except:
            pass
    else:
        flag = 0
    time.sleep(120)
