import os
import json
from datetime import datetime
from typing import Dict


def create_experiment_dir(dataset_name: str, model_name: str, root: str = 'experiments') -> str:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{ts}_{dataset_name}_{model_name}"
    exp_dir = os.path.join(root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'artifacts'), exist_ok=True)
    return exp_dir


def dump_metrics_json(metrics: Dict, exp_dir: str, filename: str = 'metrics.json') -> str:
    path = os.path.join(exp_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


