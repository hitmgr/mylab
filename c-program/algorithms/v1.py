import copy
import numpy as np
import torch
import flgo.algorithm.fedbase as fedbase
import flgo.utils.fmodule as fmodule

class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        # 初始化算法参数
        algo_params = {
            'd': self.num_clients,  # 每轮预设选取的客户端数量
            'committee_ratio': 0.3,  # 委员会大小占参与客户端的比例
            # 'alpha': 1,  # 用于计算委员会节点数 K 的参数
            'K_min': 1,  # 最小委员会节点数，确保至少有一个客户端在委员会中
            'w1': 0.5,  # 本地准确率权重
            'w2': 0.5,  # 全局准确率权重
            'gamma': 0.1,  # 用于得分计算中的时间衰减因子
            'momentum': 0.9,  # 动量因子，用于时间衰减
            'selected_round': 10,  # 每隔多少轮更新一次委员会
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
        
    def sample(self):
        """选择参与本轮训练的客户端"""
        N = min(self.d, len(self.available_clients))  # 确定本轮选择的客户端总数
        # K = max(min(N//3 + 1, self.alpha * np.log(N+1)), self.K_min)  # 计算委员会成员数量
        K = max(int(N * self.committee_ratio), self.K_min)  # 计算委员会成员数量
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
        self.gv.logger.info(f"Round {self.current_round}: Committee clients selected - {self.committee}")
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

        # 计算全局准确率
        current_global_acc = self.test(self.model)['accuracy']
        
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

class V1:
    Server = Server
    Client = Client