"""
Main driver to dispatch experiments for different algorithms. 

Algorithm name should be specified in a config file as, e.g.
```
algorithm: kaqn
```
and supported in the `imports` dispatch dict below. 

Then, this file can be invoked simply as
```
python run_experiment.py
```
"""

import hydra
from omegaconf import DictConfig

# Dispatch algorithm name to a subprocess call. We probably don't want to make
# this arbitrary (security reasons)
imports = {
    "kaqn": exec('from kaqn import main as algo'),
}


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config: DictConfig):
    imports[config.algorithm]
    algo(config)

if __name__ == "__main__":
    main()