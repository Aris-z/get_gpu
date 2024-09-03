import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import time
from transformers import LlamaForCausalLM, LlamaTokenizer
import setproctitle
setproctitle.setproctitle("~/anaconda3/envs/LLM/bin/python")

time_start = time.time()
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

size = 1024
total_time = 30 #second
time_need = False

class randomdata(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, size, size).to('cuda')
        self.len = 1
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len

class get_gpu(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size)
    def forward(self, x):
        for _ in range(1000):
            y = self.fc(self.fc(x) @ self.fc(x))

dataset = randomdata(size)

data_load = DataLoader(dataset=dataset, batch_size=8, sampler=DistributedSampler(dataset))

_ = torch.randn((16, 1024, 1024, 1024), device=device)
model = get_gpu(size).to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
for data in data_load:
    while(True):
        time_end = time.time()
        if time_need and (time_end - time_start) > total_time:
            exit()
        model(data)
