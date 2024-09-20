import os
import flgo
import flgo.algorithm.fedavg as fedavg
import flgo.algorithm.fedprox as fedprox
from algorithms.origin import OriginAlgorithm
import flgo.benchmark.cifar10_classification.model.resnet18_gn as resnet18

path = 'tasks/origin/checkpoint_test'

if not os.path.exists(path):
    os.makedirs(path)

NUM_CLIENTS = 100

configurations = {
    "demo": {'benchmark':{'name':'flgo.benchmark.cifar10_classification'},'partitioner':{'name':'IIDPartitioner', 'para':{'num_clients':NUM_CLIENTS}}},
}

# 生成任务
for task_name, config in configurations.items():
    task_path = os.path.join(path, task_name)
    if not os.path.exists(task_path):
        flgo.gen_task(config, task_path=task_path)

def get_algorithm_option(algorithm_name):
    base_option = {
        'gpu': [1,],
        'log_file': True,
        'num_rounds': 5,
        'proportion': 1.0,
        'learning_rate': 0.01,
        'num_epochs': 1,
        'sample': 'uniform',
        'eval_interval': 1,
        'save_checkpoint': 123,
        # 'load_checkpoint': 123,
    }
    
    if algorithm_name == 'OriginAlgorithm':
        base_option['algo_para'] = [
            100,  # d: 每轮预设选取的客户端数量
            0.1,  # committee_ratio: 委员会大小占参与客户端的比例
            1,  # K_min: 最小委员会节点数
            0.3,  # w_acc: 准确率提升权重
            0.3,  # w_grad: 梯度质量权重
            0.2,  # w_time: 时间衰减权重
            0.2,  # w_committee: 委员会奖励权重
            0.1,  # gamma: 时间衰减因子
            2,  # selected_round: 更新委员会的轮次间隔
            95,  # tau_percentile: 用于计算tau的百分位数
        ]
    elif algorithm_name == 'fedprox':
        base_option['algo_para'] = [
            0.1,  # mu: fedprox的mu参数
        ]
    # fedavg 不需要特殊的 algo_para

    return base_option

# 定义任务路径和划分方式
task_prefix = path
partitions = list(configurations.keys())

for partition in partitions:
    task = os.path.join(task_prefix, partition)

# 选择要运行的算法
# 可以通过修改这里的值来选择不同的算法
algorithm_name = 'fedprox'  # 可以是 'OriginAlgorithm', 'fedavg', 或 'fedprox'

# 获取对应算法的option
option = get_algorithm_option(algorithm_name)

# 获取对应的算法对象
algorithm = globals()[algorithm_name]

# 运行选定的算法
runner = flgo.init(task=task, algorithm=algorithm, option=option, model=resnet18)
runner.run()