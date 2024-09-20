import os
import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.fedprox as fedprox
from algorithms.v2 import V2
from algorithms.v1 import V1
from algorithms.simulator import ActivityHeterogeneitySimulator_basic,ActivityHeterogeneitySimulator_Committee
import flgo.experiment.device_scheduler as ds
import torch.multiprocessing
import flgo.benchmark.cifar10_classification.model.resnet18_gn as resnet18

# 设置基本路径
BASE_PATH = 'tasks/9.20'
NUM_CLIENTS = 50

# 确保基本路径存在
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# 定义不同的数据划分配置
configurations = {
    "iid": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'IIDPartitioner', 'para': {'num_clients': NUM_CLIENTS}}},
    "div01": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'DiversityPartitioner', 'para': {'num_clients': NUM_CLIENTS, 'diversity': 0.1}}},
    "div05": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'DiversityPartitioner', 'para': {'num_clients': NUM_CLIENTS, 'diversity': 0.5}}},
    "div09": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'DiversityPartitioner', 'para': {'num_clients': NUM_CLIENTS, 'diversity': 0.9}}},
    "dir01": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'DirichletPartitioner', 'para': {'num_clients': NUM_CLIENTS, 'alpha': 0.1}}},
    "dir10": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'DirichletPartitioner', 'para': {'num_clients': NUM_CLIENTS, 'alpha': 1.0}}},
    "dir50": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'DirichletPartitioner', 'para': {'num_clients': NUM_CLIENTS, 'alpha': 5.0}}}
}

# 生成任务
for task_name, config in configurations.items():
    task_path = os.path.join(BASE_PATH, task_name)
    if not os.path.exists(task_path):
        flgo.gen_task(config, task_path=task_path)
print("Tasks generated successfully under", BASE_PATH)

# 为每个算法定义特定的option
def get_algorithm_option(algorithm_name):
    base_option = {
        'log_file': True,
        'num_rounds': 20,
        'proportion': 1.0,
        'learning_rate': 0.1,  # 0.01-0.001
        # 'num_epochs': 1,
        'batch_size':64,
        # 'eval_interval': 1,
        # 'save_checkpoint': 123,
        # 'load_checkpoint': 123,
    }
    
    if algorithm_name == 'V2':
        base_option['algo_para'] = [
            NUM_CLIENTS,  # d: 每轮预设选取的客户端数量
            0.1,  # committee_ratio: 委员会大小占参与客户端的比例
            1,  # K_min: 最小委员会节点数
            0.5,  # w_acc: 准确率提升权重
            0.4,  # w_grad: 梯度质量权重
            0.05,  # w_time: 时间衰减权重
            0.05,  # w_committee: 委员会奖励权重
            0.1,  # gamma: 时间衰减因子
            10,  # selected_round: 更新委员会的轮次间隔
            95,  # tau_percentile: 用于计算tau的百分位数
        ]
    elif algorithm_name == 'fedprox':
        base_option['algo_para'] = [
            0.1,  # mu: fedprox的mu参数
        ]
    # fedavg 不需要特殊的 algo_para

    return base_option

# 根据不同的算法选择不同的simulator
def get_algorithm_simulator(algorithm_name):
    if algorithm_name == 'MyAlgorithmTime':
        simulator = ActivityHeterogeneitySimulator_Committee
    elif algorithm_name == 'fedavg' or 'fedprox':
        simulator = ActivityHeterogeneitySimulator_basic
    else:
        # 退出并报错
        print(f"Unknown algorithm: {algorithm_name}")
    
    return simulator
        

# 为每个算法和每个划分方式生成对应的Runner配置
runner_dict = []
partitions = list(configurations.keys())
algorithm_list = [
    'V1', 
    'V2',
    'fedavg', 
    'fedprox'
]

for partition in partitions:
    task = os.path.join(BASE_PATH, partition)
    for algorithm_name in algorithm_list:
        algorithm = globals()[algorithm_name]
        option = get_algorithm_option(algorithm_name)
        simulator = get_algorithm_simulator(algorithm_name)
        runner_dict.append({
            'task': task,
            'algorithm': algorithm,
            'option': option,
            # 'model': resnet18,
            # 'Simulator': simulator
        })
        
# 使用AutoScheduler，指定使用的GPU编号
asc = ds.AutoScheduler([0,1,])

if __name__ == '__main__':
    # 设置多进程启动方法和共享策略
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    # 并行运行这些Runner
    flgo.multi_init_and_run(runner_dict, scheduler=asc)