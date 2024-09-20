import os
import sys
import yaml
import flgo
import importlib
import flgo.experiment.device_scheduler as ds

from algorithms.origin import OriginAlgorithm
import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.fedprox as fedprox


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def import_algorithm(algorithm_name):
    if algorithm_name in globals():
        return globals()[algorithm_name]
    else:
        try:
            module_name = f"flgo.algorithm.{algorithm_name.lower()}"
            module = importlib.import_module(module_name)
            return getattr(module, algorithm_name)
        except ImportError:
            try:
                module_name = "algorithms.origin"
                module = importlib.import_module(module_name)
                return getattr(module, algorithm_name)
            except ImportError:
                raise ImportError(f"无法导入算法 {algorithm_name}")

def generate_tasks(config):
    base_path = config['base_path']
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for task_name, task_config in config['tasks'].items():
        task_path = os.path.join(base_path, task_name)
        if not os.path.exists(task_path):
            task_gen_config = {
                'benchmark': task_config['benchmark'],
                'partitioner': task_config['partitioner']
            }
            flgo.gen_task(task_gen_config, task_path=task_path)
    print("Tasks generated successfully under", base_path)

def generate_runners(config):
    runners = []
    base_path = config['base_path']
    for task_name, task_config in config['tasks'].items():
        task_path = os.path.join(base_path, task_name)
        algorithm = import_algorithm(task_config['algorithm'])
        runners.append({
            'task': task_path,
            'algorithm': algorithm,
            'option': task_config['option']
        })
    return runners

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")

    config = load_config('config.yaml')
    generate_tasks(config)
    runners = generate_runners(config)

    asc = ds.AutoScheduler(config['gpu_ids'])
    flgo.multi_init_and_run(runners, scheduler=asc)