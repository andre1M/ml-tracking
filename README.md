## Getting started 

Install all the dependencies form the pyproject.toml with [Poetry](https://python-poetry.org) by running the following in the project root: 
```commandline
poetry install --no-root
```
or by manually installing the packages with conda or pip.

Run the `train.py` script with the following command:
```commandline
python train.py -m model=simple_mlp,simple_conv
```

Alternatively, if using [Poetry](https://python-poetry.org), run
```commandline
poetry run python train.py -m model=simple_mlp,simple_conv
```

This will create:

- `ouputs` directory with tensorboard logs and model checkpoints per run
- `multirun` directory with runs information
- `mlruns` MLFlow directory

To explore MLFlow logs, in your terminal emulator run
```commandline
mlfow ui
```
or, if using [Poetry](https://python-poetry.org), 
```commandline
poetry run mlflow ui
```

and open the prompted link to the local host. There you'll find lots of nice visualizations, comparisons, etc.

### CUDA

For CUDA supporter version of PyTorch, one would need to handle `torch` and `torchvision` dependencies themselves ([guide](https://pytorch.org/get-started/locally/)).
