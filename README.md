# Quarto - Reinforcement Learning

This repo contains a little bit of theory around Reinforcement Learning (RL), using a 2-player game called Quarto as a practical example.

## How to Use

1. This repo uses conda to manage the Python, Jupyter Notebook and other dependencies.
If you do not have it installed yet, you just have to follow the [official instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for your platform.
2. With `conda` installed, you should clone this repo, by opening a command prompt and executing:
	```sh
	git clone https://github.com/school-of-ai-angers/quarto.git
	cd quarto
	```
3. While still inside the same terminal window, install the dependencies and start the Jupyter server:
	```sh
	conda env create
	jupyter notebook
	````
	You should now see a new tab in your Browser.

## Contents

You may start by reading a take on [Reinforcement Learning basics](./Reinforcement%20Learning.ipynb). We encourage you to also search online more examples and other topics.

Then, you can use the notebook [Play against it](./Play%20against%20it.ipynb) to manually explore the Quarto game and the environment (how the state and action spaces are encoded, rewards, etc). In doubt, the environment is implemented by [the environment.py file](./quarto/environment.py) with comments to ease understanding.

Finally, a "working" implementation of Q-learning is given in the notebook [Train your player](./Train%20your%20player.ipynb). We encourage you to read the algorithm implementation and see how the agent training run. Then, you can use this squeleton to implement your own algorithm and train your agent.
For submission in an [online plateforme - Arena](https://angers.schoolofai.fr/) to compete with others implementations, you need to create a zip file with your code. The last cell has a tool to create this zip in the right format.

## The Arena