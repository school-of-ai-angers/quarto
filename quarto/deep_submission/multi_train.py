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
    'epsilon': [0.85, 0.9, 0.95],
    'min_epsilon': [0.025, 0.05, 0.1],
    'epsilon_decay': [0.99995, 0.99999, 0.999995],
    'tau': [1e-3, 1e-4, 1e-5],
    'gamma': [0.9, 0.99, 1],
    'lr': [1e-5, 1e-6, 5e-7],
    'gradient_clip': [0.3, 1, 3],
    'hidden_size': [128, 256, 512],
}

max_minutes = 90
num = 20
per_droplet = 2
commit = '4af55e5a4ccba9d98649fdd244186a5ba56059bc'

commands = []
for i in range(num):
    concrete_params = {
        'max_minutes': max_minutes,
        'name': f'player2-{i:02d}'
    }
    for key, values in params.items():
        concrete_params[key] = random.choice(values)

    command_args = ' '.join(f'--{key} {value}' for key, value in concrete_params.items())
    command = f'-m quarto.deep_submission.train {command_args} &>{concrete_params["name"]}.log'
    commands.append(command)

for i in range(int(num / per_droplet)):
    droplet_commands = '\n'.join(
        f'docker run -v "$PWD/weights:/app/weights" quarto {command}'
        for command in commands[i * per_droplet:(i+1)*per_droplet]
    )

    droplet_name = f'quarto-train-{i:02d}'
    user_data = f'''#!/bin/bash -e
    cd /root/quarto
    git fetch
    git checkout --force {commit}
    docker build -t quarto -f quarto/deep_submission/Dockerfile .
    {droplet_commands}
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
