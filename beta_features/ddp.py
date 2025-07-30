import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test on single GPU')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=2048, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--world_size', type=int, default=68, help='number of processes to simulate DDP')
parser.add_argument('--master_port', type=str, default='12355', help='master port for communication')

def setup(rank, world_size, master_port):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.3'
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize the process group
    # Using 'gloo' backend since we're simulating multiple processes on same GPU
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run_training(rank, world_size, args):
    print(f"Running DDP training on rank {rank} of {world_size}")
    
    # Setup the process group
    setup(rank, world_size, args.master_port)
    
    # Set device - all processes use the same GPU (GPU 0)
    device = 'cuda:1'
    
    print(f'From Rank: {rank}, ==> Making model..')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create model and move to device
    net = Net().to(device)
    
    # Wrap model with DDP
    net = torch.nn.parallel.DistributedDataParallel(net)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    print(f'From Rank: {rank}, ==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = CIFAR10(root='datasets', train=True, download=True, transform=transform_train)

    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset_train, 
        num_replicas=world_size, 
        rank=rank
    )
    
    train_loader = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=False,  # Don't shuffle when using DistributedSampler
        num_workers=args.num_workers, 
        sampler=train_sampler,
        pin_memory=True
    )

    perf = []
    total_start = time.time()
    
    print(f"Rank {rank}: Starting training with {len(train_loader)} batches")

    for _ in range(100):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            start = time.time()
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - start
            images_per_sec = args.batch_size / batch_time
            perf.append(images_per_sec)
            
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Batch {batch_idx}, Loss: {loss.item():.4f}, Images/sec: {images_per_sec:.2f}")

        total_time = time.time() - total_start
        avg_perf = np.mean(perf)
    
    print(f"Rank {rank}: Training completed in {total_time:.2f}s, Avg performance: {avg_perf:.2f} images/sec")
    
    # Clean up
    cleanup()

def main():
    print("Starting DDP training on single GPU...")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
    else:
        print(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
    
    world_size = args.world_size
    
    # Spawn processes for DDP
    mp.spawn(
        run_training,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

def f(x):
    print(x)

if __name__ == '__main__':
    # main()



    mp.spawn(
    f,
    nprocs=10,
    join=True
    )
