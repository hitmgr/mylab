import os
import flgo
from algorithms.my_algorithm import MyAlgorithm
from algorithms.my_algorithm_adjust_weight import MyAlgorithmAdjustWeight
import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.fedprox as fedprox
import flgo.experiment.device_scheduler as ds

path = '/home/mgr/sss/easyFL-FLGo/a-program/tasks/9.20'
if not os.path.exists(path):
    os.makedirs(path)

# 定义常量
NUM_CLIENTS = 100

# 定义不同数据异构性配置，并将任务路径放在 tasks/multi-data-2/ 目录下
configurations = {
    "iid": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'IIDPartitioner', 'para':{'num_clients': NUM_CLIENTS}}},
    # "div01": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'DiversityPartitioner', 'para':{'num_clients': NUM_CLIENTS, 'diversity':0.1}}},
    # "div05": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'DiversityPartitioner', 'para':{'num_clients': NUM_CLIENTS, 'diversity':0.5}}},
    # "div09": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'DiversityPartitioner', 'para':{'num_clients': NUM_CLIENTS, 'diversity':0.9}}},
    # "dir01": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'DirichletPartitioner', 'para':{'num_clients': NUM_CLIENTS, 'alpha':0.1}}},
    # "dir10": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'DirichletPartitioner', 'para':{'num_clients': NUM_CLIENTS, 'alpha':1.0}}},
    # "dir50": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'DirichletPartitioner', 'para':{'num_clients': NUM_CLIENTS, 'alpha':5.0}}}
}

# configurations = {
#     "div01-patience10-adjust_rate0.005": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'}, 'partitioner':{'name':'DiversityPartitioner', 'para':{'num_clients': NUM_CLIENTS, 'diversity':0.1}}},
# }

# 遍历配置，生成相应的任务
for task_name, config in configurations.items():
    task_path = os.path.join(path, task_name)
    if not os.path.exists(task_path):
        flgo.gen_task(config, task_path=task_path)

print("Tasks generated successfully under", path)

# 定义任务路径和划分方式
task_prefix = path
partitions = list(configurations.keys())

# 定义通用的Runner选项
option = {'log_file': True, 'num_rounds': 200, 'proportion': 1.0, 'learning_rate': 0.008, 'num_epochs': 1, 'sample': 'uniform'}

# 为每个算法和每个划分方式生成对应的Runner配置
runner_dict = []

for partition in partitions:
    task = os.path.join(task_prefix, partition)
    
    # algorithm_list = ['MyAlgorithm', 'fedavg', 'fedprox']
    algorithm_list = [
        'MyAlgorithm',
        # 'fedavg' 
        ]

    for algorithm_name in algorithm_list:
        algorithm = globals()[algorithm_name]  # 将字符串转换为函数或类对象
        runner_dict.append({'task': task, 'algorithm': algorithm, 'option': option})

# 使用AutoScheduler，指定使用的GPU编号为1
asc = ds.AutoScheduler([0,1])

# 并行运行这些Runner
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    flgo.multi_init_and_run(runner_dict, scheduler=asc)