# bipedal-walker-deep-ga
The project solves *Bipedal-Walker-v3* using deep neural networks trained by genetic algorithm. We used simplified version
of [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567). 
Our best model reached average return of ~310 over 100 episodes.

![Model](docs/model.gif)

## How to install project
1. Clone the repository and change current directory
```shell script
git clone https://github.com/standa42/bipedal-walker-deep-ga
cd "bipedal-walker-deep-ga"
```

2. Create virtual environment
- you can also reuse an existing one but this tutorial acts as you would create a new one

```shell script
/usr/bin/python3 -m venv "venv"
```

3. Install pip requirements
```shell script
venv/bin/python3 -m pip install -r "requirements.txt"
```

## How to train a model
Training is simple using following script:
```shell script
/venv/bin/python3 train.py
```

Default parameters are already set for the best reached model.
Logs are stored into *logs/train{TIMESTAMP}_{UUID4_CODE}* directory.

## How to evaluate a model
Evaluation of model is done using following script:
```shell script
/venv/bin/python3 evaluate.py
```

Evaluation can be visualized using *render_each* parameter.