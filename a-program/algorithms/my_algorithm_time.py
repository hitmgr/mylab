import copy
import numpy as np
import torch
import flgo.algorithm.fedbase as fedbase
import flgo.utils.fmodule as fmodule
import flgo.simulator.base
import random

class ActivityHeterogeneitySimulator_Committee(flgo.simulator.base.BasicSimulator):
    def __init__(self, objects, *args, **kwargs):
        super().__init__(objects, *args, **kwargs)
        self.committee = []  # 存储委员会成员

    def update_client_availability(self):
        if self.gv.clock.current_time == 0:
            self.set_variable(self.all_clients, 'prob_available', [1 for _ in self.clients])
            self.set_variable(self.all_clients, 'prob_unavailable', [int(random.random() >= 0.5) for _ in self.clients])
            return

        # 更新委员会成员列表
        self.committee = self.server.committee

        # 低频率活跃性异构
        pa = [0.1 for _ in self.clients]
        pua = [0.1 for _ in self.clients]

        # 高频率活跃性异构（注释掉）
        # pa = [0.9 for _ in self.clients]
        # pua = [0.9 for _ in self.clients]

        # 确保委员会成员始终可用
        for cid in self.all_clients:
            if cid in self.committee:
                self.variables[cid]['prob_available'] = 1.0
                self.variables[cid]['prob_unavailable'] = 0.0
            else:
                self.variables[cid]['prob_available'] = pa[cid]
                self.variables[cid]['prob_unavailable'] = pua[cid]

        # 打印日志
        self.gv.logger.info(f"Round {self.gv.clock.current_time}: Committee members: {self.committee}")
        self.gv.logger.info(f"Round {self.gv.clock.current_time}: Availability probabilities updated")
        self.gv.logger.write_var_into_output(f"committee_round_{self.gv.clock.current_time}", list(self.committee))
        
        # 记录更新后的可用性概率
        updated_pa = [self.variables[cid]['prob_available'] for cid in self.all_clients]
        updated_pua = [self.variables[cid]['prob_unavailable'] for cid in self.all_clients]
        self.gv.logger.write_var_into_output(f"availability_probs_round_{self.gv.clock.current_time}", 
                                             {"pa": updated_pa, "pua": updated_pua})

class ActivityHeterogeneitySimulator_fedavg(flgo.simulator.base.BasicSimulator):
    def update_client_availability(self):
        if self.gv.clock.current_time==0:
            self.set_variable(self.all_clients, 'prob_available', [1 for _ in self.clients])
            self.set_variable(self.all_clients, 'prob_unavailable', [int(random.random() >= 0.5) for _ in self.clients])
            return
        pa = [0.1 for _ in self.clients]
        pua = [0.1 for _ in self.clients]
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)

class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        # 初始化算法参数
        algo_params = {
            'd': self.num_clients,  # 每轮预设选取的客户端数量
            'alpha': 1,  # 用于计算委员会节点数 K 的参数
            'K_min': 1,  # 最小委员会节点数，确保至少有一个客户端在委员会中
            'w1': 0.3,  # 初始本地准确率权重
            'w2': 0.7,  # 初始全局准确率权重
            'gamma': 0.1,  # 用于得分计算中的时间衰减因子
            'momentum': 0.9,  # 动量因子，用于时间衰减
            'selected_round': 25,  # 每隔多少轮更新一次委员会
            'adjust_rate': 0.005,  # 自适应权重调整速率
            'patience': 10,  # 耐心轮数，用于判断全局准确率的变化
        }
        self.init_algo_para(algo_params)
        self.gv.logger.info(f"Initialization parameters: {algo_params}")
        self.gv.logger.write_var_into_output("initialization_parameters", algo_params)
        
        # 初始化客户端的参与记录、时间衰减因子和其他变量
        self.last_participation = {cid: 0 for cid in range(self.num_clients)}  # 记录每个客户端最后一次参与的轮次
        self.time_decay_factors = {cid: 1.0 for cid in range(self.num_clients)}  # 初始化时间衰减因子
        self.prev_global_acc = 0.0  # 存储上一轮全局准确率
        self.stable_rounds = 0  # 记录全局准确率稳定的轮数
        self.committee = []  # 存储当前委员会成员
        self.scores = {cid: 0.0 for cid in range(self.num_clients)}  # 存储每个客户端的得分
        
    def adjust_weights(self, current_global_acc):
        """自适应调整本地和全局准确率的权重"""
        old_w1, old_w2 = self.w1, self.w2  # 保存调整前的权重

        if current_global_acc > self.prev_global_acc:
            self.stable_rounds += 1
            if self.stable_rounds >= self.patience:
                # 如果全局准确率连续上升，增加全局准确率的权重
                self.w2 = min(1.0, self.w2 + self.adjust_rate)                    
                self.w1 = 1.0 - self.w2
                self.stable_rounds = 0  # 重置稳定轮数
        else:
            # 如果全局准确率下降，增加本地准确率的权重
            self.stable_rounds = 0
            self.w1 = min(1.0, self.w1 + self.adjust_rate)
            self.w2 = 1.0 - self.w1

        self.prev_global_acc = current_global_acc  # 更新上一轮全局准确率

        # 打印和记录权重变化
        print(f"Round {self.current_round}: Weights adjusted - w1: {old_w1:.4f} -> {self.w1:.4f}, w2: {old_w2:.4f} -> {self.w2:.4f}")
        self.gv.logger.info(f"Round {self.current_round}: Weights adjusted - w1: {old_w1:.4f} -> {self.w1:.4f}, w2: {old_w2:.4f} -> {self.w2:.4f}")
        self.gv.logger.write_var_into_output(f"weights_round_{self.current_round}", {
            "w1_old": float(old_w1),
            "w1_new": float(self.w1),
            "w2_old": float(old_w2),
            "w2_new": float(self.w2)
        })

    def sample(self):
        """选择参与本轮训练的客户端"""
        N = min(self.d, len(self.available_clients))  # 确定本轮选择的客户端总数
        K = max(min(N//3 + 1, self.alpha * np.log(N+1)), self.K_min)  # 计算委员会成员数量
        K = int(K)

        # 更新委员会成员（如果需要）
        if not self.committee or self.current_round % self.selected_round == 0:
            sorted_clients = sorted(self.available_clients, key=lambda x: self.scores[x], reverse=True)
            self.committee = sorted_clients[:K]
            self.gv.logger.info(f"Round {self.current_round}: Committee updated - {self.committee}")
            self.gv.logger.write_var_into_output(f"committee_round_{self.current_round}", list(self.committee))

        # 从非委员会成员中随机选择普通客户端
        normal_clients = [cid for cid in self.available_clients if cid not in self.committee]
        selected_normal = np.random.choice(normal_clients, min(N-K, len(normal_clients)), replace=False)
        selected_normal = list(map(int, selected_normal))
        selected_normal.sort()

        # 合并委员会成员和随机选择的普通客户端
        selected_clients = list(self.committee) + selected_normal
        selected_clients.sort()

        # 记录选择的客户端信息
        self.gv.logger.info(f"Round {self.current_round}: Committe clients selected - {self.committee}")
        self.gv.logger.write_var_into_output(f"committee_clients_round_{self.current_round}", list(self.committee))
        self.gv.logger.info(f"Round {self.current_round}: Normal clients selected - {selected_normal}")
        self.gv.logger.write_var_into_output(f"normal_clients_round_{self.current_round}", selected_normal)
        self.gv.logger.info(f"Round {self.current_round}: All selected clients - {selected_clients}")
        self.gv.logger.write_var_into_output(f"selected_clients_round_{self.current_round}", selected_clients)
        return selected_clients

    def iterate(self):
        """每一轮训练和得分计算的过程"""
        self.gv.logger.info(f"Round {self.current_round} started")
        self.gv.logger.write_var_into_output(f"round_start", self.current_round)

        # 在每轮开始时记录当前的w1和w2
        self.gv.logger.info(f"Round {self.current_round}: Current weights - w1: {self.w1:.4f}, w2: {self.w2:.4f}")
        self.gv.logger.write_var_into_output(f"weights_start_round_{self.current_round}", {
            "w1": float(self.w1),
            "w2": float(self.w2)
        })

        self.selected_clients = self.sample()  # 选择参与本轮训练的客户端
        res = self.communicate(self.selected_clients)  # 与选中的客户端进行通信
        models = res['model']  # 获取客户端更新后的模型

        # 计算全局准确率并自适应调整权重
        current_global_acc = self.test(self.model)['accuracy']
        self.adjust_weights(current_global_acc)

        raw_scores = {}  # 存储客户端的原始得分

        for cid, model in zip(self.selected_clients, models):
            local_acc = self.clients[cid].test(model)['accuracy']  # 计算本地准确率
            grad = fmodule._model_sub(self.model, model)  # 计算模型梯度
            grad_norm = fmodule._model_norm(grad)  # 计算梯度范数

            # 动量式时间衰减
            time_decay = self.momentum * self.time_decay_factors[cid] + (1 - self.momentum) * np.exp(-self.gamma * (self.current_round - self.last_participation[cid]))
            self.time_decay_factors[cid] = time_decay  # 更新时间衰减因子

            # 计算客户端原始得分
            raw_score = (self.w1 * local_acc + self.w2 * current_global_acc) * (1 / (1 + grad_norm.item())) * time_decay
            raw_scores[cid] = float(raw_score)
            self.last_participation[cid] = self.current_round  # 更新客户端最后参与轮次

        # 归一化客户端得分
        if len(raw_scores) > 1:
            min_score = min(raw_scores.values())
            max_score = max(raw_scores.values())
            if max_score > min_score:
                for cid in raw_scores:
                    self.scores[cid] = (raw_scores[cid] - min_score) / (max_score - min_score)
            else:
                for cid in raw_scores:
                    self.scores[cid] = 1.0
        else:
            for cid in raw_scores:
                self.scores[cid] = 1.0

        # 记录客户端信息，包括归一化后的得分
        for cid in self.selected_clients:
            local_acc = self.clients[cid].test(models[self.selected_clients.index(cid)])['accuracy']
            grad_norm = fmodule._model_norm(fmodule._model_sub(self.model, models[self.selected_clients.index(cid)]))
            info_str = f"Round {self.current_round}, Client {cid}: Local Acc={local_acc:.4f}, Global Acc={current_global_acc:.4f}, Grad Norm={grad_norm:.4f}, Normalized Score={self.scores[cid]:.4f}"
            self.gv.logger.info(info_str)
            self.gv.logger.write_var_into_output(f"client_{cid}_metrics_round_{self.current_round}", {
                "local_acc": float(local_acc),
                "global_acc": float(current_global_acc),
                "grad_norm": float(grad_norm.item()),
                "normalized_score": float(self.scores[cid])
            })

        # 聚合模型
        self.model = self.aggregate(models)
        return True

    def aggregate(self, models):
        """基于得分计算聚合权重，并更新全局模型"""
        total_score = sum([self.scores[cid] for cid in self.selected_clients])
        weights = [(self.scores[cid] / total_score) for cid in self.selected_clients]

        # 记录每个客户端的聚合权重
        for cid, weight in zip(self.selected_clients, weights):
            self.gv.logger.info(f"Round {self.current_round}, Client {cid}: Aggregation Weight={weight:.4f}")
            self.gv.logger.write_var_into_output(f"client_{cid}_aggregation_weight_round_{self.current_round}", float(weight))

        self.gv.logger.info(f"Round {self.current_round}: Total Aggregation Weight={sum(weights):.4f}")
        self.gv.logger.write_var_into_output(f"total_aggregation_weight_round_{self.current_round}", float(sum(weights)))

        return fmodule._model_average(models, weights)  # 使用计算得到的权重进行加权平均

    
class Client(fedbase.BasicClient):
    def train(self, model):
        # 使用父类的train方法进行本地训练
        return super().train(model)

class MyAlgorithmTime:
    Server = Server
    Client = Client