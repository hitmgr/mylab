import copy
import numpy as np
import torch
import flgo.algorithm.fedbase as fedbase
import flgo.utils.fmodule as fmodule

class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        # 初始化算法参数，包括客户端数量、控制参数、梯度影响因子等
        algo_params = {
            'd': self.num_clients,  # 每轮预设选取的客户端数量
            'alpha': 1,  # 用于计算委员会节点数 K 的参数
            'K_min': 1,  # 最小委员会节点数，确保至少有一个客户端在委员会中
            'delta_grad': 0.1,  # 用于委员会节点得分计算中的梯度惩罚因子
            'gamma': 0.1,  # 用于得分计算中的时间衰减因子
            'w_data': 0.5,  # 数据量权重，用于聚合模型时的加权（注：此参数在当前代码中未使用）
            'selected_round': 1,  # 每隔多少轮更新一次委员会
        }
        self.init_algo_para(algo_params)
        # 记录初始化参数到日志
        self.gv.logger.info(f"Initialization parameters: {algo_params}")
        self.gv.logger.write_var_into_output("initialization_parameters", algo_params)

        # 初始化委员会列表和客户端的参与记录与得分记录
        self.committee = []  # 当前委员会成员列表
        self.last_participation = {cid: 0 for cid in range(self.num_clients)}  # 记录每个客户端上次参与的轮次
        self.scores = {cid: 0.0 for cid in range(self.num_clients)}  # 记录每个客户端的得分
        
    def sample(self):
        # 计算本轮参与训练的客户端总数N和委员会大小K
        N = min(self.d, len(self.available_clients))
        K = max(min(N//3 + 1, self.alpha * np.log(N+1)), self.K_min)
        K = int(K)

        # 每隔 selected_round 轮或首次运行时，基于得分排序选择委员会成员
        if not self.committee or self.current_round % self.selected_round == 0:
            sorted_clients = sorted(self.available_clients, key=lambda x: self.scores[x], reverse=True)
            self.committee = sorted_clients[:K]
            # 记录委员会成员更新信息
            self.gv.logger.info(f"Round {self.current_round}: Committee updated - {self.committee}")
            self.gv.logger.write_var_into_output(f"committee_round_{self.current_round}", list(self.committee))

        # 从非委员会成员中随机选择普通客户端，确保参与总数为N
        normal_clients = [cid for cid in self.available_clients if cid not in self.committee]
        selected_normal = np.random.choice(normal_clients, min(N-K, len(normal_clients)), replace=False)
        selected_normal = list(map(int, selected_normal))  # 确保是整数列表
        selected_normal.sort()

        # 合并委员会成员和随机选择的普通客户端，形成最终的参与客户端列表
        selected_clients = list(self.committee) + selected_normal
        selected_clients.sort()

        # 记录普通客户端和最终选择的客户端列表
        self.gv.logger.info(f"Round {self.current_round}: Normal clients selected - {selected_normal}")
        self.gv.logger.write_var_into_output(f"normal_clients_round_{self.current_round}", selected_normal)
        self.gv.logger.info(f"Round {self.current_round}: All selected clients - {selected_clients}")
        self.gv.logger.write_var_into_output(f"selected_clients_round_{self.current_round}", selected_clients)
        return selected_clients

    def iterate(self):
        # 记录当前轮次开始
        self.gv.logger.info(f"Round {self.current_round} started")
        self.gv.logger.write_var_into_output(f"round_start", self.current_round)

        # 选择参与本轮训练的客户端
        self.selected_clients = self.sample()
        # 与选中的客户端进行通信，获取训练结果
        res = self.communicate(self.selected_clients)
        models = res['model']

        # 遍历每个选中的客户端，计算得分并更新相关信息
        for cid, model in zip(self.selected_clients, models):
            # 计算本地和全局准确率
            local_acc = self.clients[cid].test(model)['accuracy']
            global_acc = self.test(self.model)['accuracy']

            # 计算梯度和梯度范数
            grad = fmodule._model_sub(self.model, model)
            grad_norm = fmodule._model_norm(grad)
            smoothed_grad_norm = torch.log(1 + grad_norm)

            # 计算时间衰减因子
            time_decay = np.exp(-self.gamma * (self.current_round - self.last_participation[cid]))

            # 根据客户端是否在委员会中，使用不同的得分计算方法
            if cid in self.committee:
                base_score = global_acc * (1 - self.delta_grad * smoothed_grad_norm.item())
            else:
                # 使用委员会模型测试普通节点得分
                Acc_min = min([self.clients[i].test(models[i])['accuracy'] for i in range(len(models))])
                base_score = max(0, local_acc - Acc_min)

            # 更新客户端得分和最后参与轮次
            self.scores[cid] = float(base_score * time_decay)
            self.last_participation[cid] = self.current_round

            # 记录客户端相关指标
            info_str = f"Round {self.current_round}, Client {cid}: Local Acc={local_acc:.4f}, Global Acc={global_acc:.4f}, Grad Norm={grad_norm:.4f}, Smoothed Grad Norm={smoothed_grad_norm:.4f}, Time Decay={time_decay:.4f}, Score={self.scores[cid]:.4f}"
            self.gv.logger.info(info_str)
            self.gv.logger.write_var_into_output(f"client_{cid}_metrics_round_{self.current_round}", {
                "local_acc": float(local_acc),
                "global_acc": float(global_acc),
                "grad_norm": float(grad_norm.item()),
                "smoothed_grad_norm": float(smoothed_grad_norm.item()),
                "time_decay": float(time_decay),
                "score": float(self.scores[cid])
            })

        # 聚合模型
        self.model = self.aggregate(models)
        return True

    def aggregate(self, models):
        # 计算总得分和每个客户端的权重
        total_score = sum([self.scores[cid] for cid in self.selected_clients])
        weights = [(self.scores[cid] / total_score) for cid in self.selected_clients]

        # 记录每个客户端的聚合权重
        for cid, weight in zip(self.selected_clients, weights):
            self.gv.logger.info(f"Round {self.current_round}, Client {cid}: Aggregation Weight={weight:.4f}")
            self.gv.logger.write_var_into_output(f"client_{cid}_aggregation_weight_round_{self.current_round}", float(weight))

        # 记录总聚合权重
        self.gv.logger.info(f"Round {self.current_round}: Total Aggregation Weight={sum(weights):.4f}")
        self.gv.logger.write_var_into_output(f"total_aggregation_weight_round_{self.current_round}", float(sum(weights)))

        # 使用加权平均聚合模型
        return fmodule._model_average(models, weights)

class Client(fedbase.BasicClient):
    def train(self, model):
        # 调用父类的训练方法
        return super().train(model)

class MyAlgorithm:
    Server = Server
    Client = Client