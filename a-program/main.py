import os
import flgo
from algorithms.my_algorithm_time import MyAlgorithmTime
import flgo.algorithm.fedavg as fedavg
from algorithms.my_algorithm_time import ActivityHeterogeneitySimulator_Committee
from algorithms.my_algorithm_time import ActivityHeterogeneitySimulator_fedavg


from experiments.analyzer import analyze
from experiments.analyzer_and_painter import AnalyzerAndPainter
from flgo.experiment.logger import BasicLogger


task = './tasks/timeAvgTry'

gen_config = {
    'benchmark':{'name':'flgo.benchmark.cifar10_classification'},
    'partitioner':{'name':'IIDPartitioner', 'para':{'num_clients':100}}
}

# 生成任务
if not os.path.exists(task):
    flgo.gen_task(gen_config, task_path=task)

option = {'gpu':[1,],'log_file':True, 'num_rounds':30, 'proportion':1.0, 'learning_rate':0.01, 'num_epochs':1, 'sample':'uniform'}

# 运行自定义算法
MyAlgorithm_runner = flgo.init(task, fedavg, option=option,Simulator=ActivityHeterogeneitySimulator_fedavg)
MyAlgorithm_runner.run()

# # 分析结果
# # analyze(task)

# # 初始化analyzer_and_painter类
# ap = AnalyzerAndPainter()

# # 调用分析和绘图函数
# rounds, local_acc, global_acc, grad_norm, time_decay, score, agg_weight, committee_counts, normal_client_counts = ap.analyze_and_plot(task, option)

# # 绘制图像
# ap.plot_accuracy(rounds, local_acc, global_acc, list(local_acc.keys()))
# ap.plot_grad_time_decay(rounds, grad_norm, time_decay, list(grad_norm.keys()))
# ap.plot_score_agg_weight(rounds, score, agg_weight, list(score.keys()))
# ap.plot_committee_normal_clients(rounds, committee_counts, normal_client_counts)