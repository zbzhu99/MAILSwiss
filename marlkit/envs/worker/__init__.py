"""
The implementation of these env workers is based on [tianshou](https://github.com/thu-ml/tianshou/blob/master/tianshou/env/venvs.py)
"""

from marlkit.envs.worker.base import EnvWorker
from marlkit.envs.worker.dummy import DummyEnvWorker
from marlkit.envs.worker.subproc import SubprocEnvWorker

__all__ = [
    "EnvWorker",
    "SubprocEnvWorker",
    "DummyEnvWorker",
]
