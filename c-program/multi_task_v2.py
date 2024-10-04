import os
import flgo
import flgo.experiment.device_scheduler as ds
from algorithms.v2 import V2
from algorithms.simulator import ActivityHeterogeneitySimulator_Committee
import torch.multiprocessing

# 设置基本路径
BASE_PATH = 'tasks/10.3/v2_hypermeters_time0.9'
NUM_CLIENTS = 50

# 确保基本路径存在
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# 定义不同的数据划分配置
configurations = {
    "iid": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'IIDPartitioner', 'para': {'num_clients': NUM_CLIENTS}}},
    "dir50": {'benchmark': {'name': 'flgo.benchmark.cifar10_classification'}, 'partitioner': {'name': 'DirichletPartitioner', 'para': {'num_clients': NUM_CLIENTS, 'alpha': 5.0}}}
}

# 生成任务
for task_name, config in configurations.items():
    task_path = os.path.join(BASE_PATH, task_name)
    if not os.path.exists(task_path):
        flgo.gen_task(config, task_path=task_path)
print("Tasks generated successfully under", BASE_PATH)

# # 定义五种权重组合
# weight_combinations = [
#     [1, 0, 0, 0],  # 只考虑准确率提升
#     [0, 1, 0, 0],  # 只考虑梯度质量
#     [0, 0, 1, 0],  # 只考虑时间衰减
#     [0, 0, 0, 1],  # 只考虑委员会奖励
#     [0.3, 0.3, 0.2, 0.2],  # 综合考虑所有因素
# ]

weight_combinations = [
    [0.1, 0.1, 0.7, 0.1],  # Highest time decay weight
    [0.15, 0.15, 0.5, 0.2],  # High time decay weight
    [0.2, 0.2, 0.4, 0.2],  # Medium time decay weight
    [0.25, 0.25, 0.3, 0.2],  # Low time decay weight
    [0.3, 0.3, 0.2, 0.2],  # Lowest time decay weight
]

labels = [
    'Time decay dominant (0.7)',
    'High time decay weight (0.5)',
    'Balanced, medium time decay (0.4)',
    'Lower time decay weight (0.3)',
    'Minimized time decay (0.2)'
]

# 定义V2算法的基本配置
v2_base_option = {
    'log_file': True,
    'num_rounds': 100,
    'proportion': 1.0,
    'learning_rate': 0.01,
    'num_epochs': 1,
    'algo_para': [
        NUM_CLIENTS,  # d: 每轮预设选取的客户端数量
        0.1,  # committee_ratio: 委员会大小占参与客户端的比例
        1,  # K_min: 最小委员会节点数
        None,  # w_acc: 准确率提升权重（将在循环中设置）
        None,  # w_grad: 梯度质量权重（将在循环中设置）
        None,  # w_time: 时间衰减权重（将在循环中设置）
        None,  # w_committee: 委员会奖励权重（将在循环中设置）
        0.1,  # gamma: 时间衰减因子
        10,  # selected_round: 更新委员会的轮次间隔
        95,  # tau_percentile: 用于计算tau的百分位数
    ]
}

# 使用AutoScheduler，指定使用的GPU编号
asc = ds.AutoScheduler([0, 1])

if __name__ == '__main__':
    # 设置多进程启动方法和共享策略
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    
    # 创建用于并行运行的所有runner配置列表
    all_runners = []

    # 对每个数据划分配置进行实验
    for partition in configurations.keys():
        task = os.path.join(BASE_PATH, partition)
        
        print(f"Preparing experiments for {partition} partition...")
        
        # 为每个权重组合创建runner配置
        for i, weights in enumerate(weight_combinations):
            # 更新权重配置
            v2_option = v2_base_option.copy()
            v2_option['algo_para'] = v2_option['algo_para'].copy()
            v2_option['algo_para'][3:7] = weights
            
            # 添加runner配置到 all_runners 列表中
            all_runners.append({
                'task': task,
                'algorithm': V2,
                'option': v2_option,
                'Simulator': ActivityHeterogeneitySimulator_Committee
            })
    
    # 并行运行所有配置
    results = flgo.multi_init_and_run(all_runners, scheduler=asc)
    
    # 输出结果
    for i, result in enumerate(results):
        weights = weight_combinations[i % len(weight_combinations)]  # Modulo to map back to the correct weight combination
        partition = list(configurations.keys())[i // len(weight_combinations)]  # Determine partition based on index
        print(f"  Experiment {i+1} ({partition} partition): weights = {weights}")
        print(f"    Final accuracy: {result['test_accuracy'][-1]}")
        print(f"    Best accuracy: {max(result['test_accuracy'])}")
    
    print("\n")