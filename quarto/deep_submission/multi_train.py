import subprocess
import random
import digitalocean
import os
from dotenv import load_dotenv

load_dotenv()

region = os.environ['DO_REGION']
size = os.environ['DO_SIZE']
image = os.environ['DO_IMAGE']
token = os.environ['DO_TOKEN']
ssh_keys = os.environ['DO_SSH_KEYS']

params = {
    'memory_size': [int(1e4), int(1e5), int(1e6)],
    'warm_up': [int(1e2), int(1e3), int(1e4)],
    'batch_size': [32, 64, 128],
    'train_every': [4, 8, 16],
    'epsilon': [0.8, 0.9, 1],
    'min_epsilon': [0.05, 0.1, 0.15],
    'epsilon_decay': [0.99995, 0.99999, 0.999995],
    'tau': [1e-3, 1e-4, 1e-5],
    'gamma': [0.9, 0.99, 1],
    'lr': [1e-4, 1e-5, 1e-6],
    'gradient_clip': [0.3, 1, 3],
    'hidden_size': [128, 256, 512],
}

max_minutes = 60
num = 10
commit = 'dfbb67ddf958889cc9d652cf338e98af8a2f9a39'

for i in range(num):
    concrete_params = {
        'max_minutes': max_minutes,
        'name': f'player-{i:02d}'
    }
    for key, values in params.items():
        concrete_params[key] = random.choice(values)

    command_args = ' '.join(f'--{key} {value}' for key, value in concrete_params.items())
    command = f'-m quarto.deep_submission.train {command_args} &>log.log'

    droplet_name = 'quarto-train-' + concrete_params['name']
    user_data = f'''#!/bin/bash -e
    cd /root/quarto
    git fetch
    git checkout --force {commit}
    docker build -t quarto -f quarto/deep_submission/Dockerfile .
    docker run -v "$PWD/weights:/app/weights" quarto {command}
    '''
    droplet = digitalocean.Droplet(
        token=token,
        name=droplet_name,
        region=region,
        size=size,
        image=image,
        ssh_keys=ssh_keys.split(','),
        user_data=user_data,
        tags=['quarto-train']
    )
    droplet.create()
    print(f'Created droplet {droplet.id}')
