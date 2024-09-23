import numpy as np
import flgo.algorithm.fedbase as fedbase
import flgo.utils.fmodule as fmodule
import torch


class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        # 初始化算法参数
        algo_params = {
            'd': self.num_clients,  # 每轮预设选取的客户端数量
            'committee_ratio': 0.3,  # 委员会大小占参与客户端的比例
            'K_min': 1,  # 最小委员会节点数
            'w_acc': 0.3,  # 准确率提升权重
            'w_grad': 0.3,  # 梯度质量权重
            'w_time': 0.2,  # 时间衰减权重
            'w_committee': 0.2,  # 委员会奖励权重
            'gamma': 0.1,  # 时间衰减因子
            'selected_round': 10,  # 更新委员会的轮次间隔
            'tau_percentile': 95,  # 用于计算tau的百分位数
        }
        self.init_algo_para(algo_params)
        self.gv.logger.info(f"Initialization parameters: {algo_params}")
        self.gv.logger.write_var_into_output("initialization_parameters", algo_params)
        
        # 初始化客户端相关变量
        self.last_participation = {cid: 0 for cid in range(self.num_clients)}  # 记录每个客户端最后参与的轮次
        self.committee = []  # 当前委员会成员列表
        self.scores = {cid: [] for cid in range(self.num_clients)}  # 记录每个客户端所有轮次的得分
        self.grad_norm_history = []  # 记录梯度范数历史
        self.tau = 1.0  # 初始化tau值，用于梯度裁剪
        
    def sample(self):
        """选择参与本轮训练的客户端"""
        N = min(self.d, len(self.available_clients))  # 确定本轮选择的客户端总数
        K = max(int(N * self.committee_ratio), self.K_min)  # 计算委员会成员数量

        # 如果需要更新委员会
        if not self.committee or self.current_round % self.selected_round == 0:
            # 使用平均历史得分进行排序
            avg_scores = {cid: sum(scores) / len(scores) if scores else 0 
                          for cid, scores in self.scores.items()}
            sorted_clients = sorted(self.available_clients, 
                                    key=lambda x: avg_scores[x], 
                                    reverse=True)
            self.committee = sorted_clients[:K]
            self.gv.logger.info(f"Round {self.current_round}: Committee updated - {self.committee}")
            self.gv.logger.write_var_into_output(f"committee_round_{self.current_round}", list(self.committee))

        # 从非委员会成员中随机选择普通客户端
        normal_clients = [cid for cid in self.available_clients if cid not in self.committee]
        # 选择客户端，确保数量不超过剩余客户端总数
        selected_normal = np.random.choice(normal_clients, min(N-K, len(normal_clients)), replace=False)
        selected_normal = list(map(int, selected_normal))
        selected_normal.sort()

        # 合并委员会成员和随机选择的普通客户端
        selected_clients = list(self.committee) + selected_normal
        selected_clients.sort()

        self.gv.logger.info(f"Round {self.current_round}: All selected clients - {selected_clients}")
        self.gv.logger.write_var_into_output(f"selected_clients_round_{self.current_round}", selected_clients)
        return selected_clients
    
   
    def calculate_cosine_similarity(self, model1, model2):
        """计算两个模型的余弦相似度"""
        vec1, vec2 = [], []
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            vec1.append(p1.data.view(-1))
            vec2.append(p2.data.view(-1))
        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)
        return torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm() + 1e-8)

    def calculate_client_scores(self, selected_clients, models, current_global_acc):
        """计算客户端得分"""
        raw_scores = {'A': {}, 'G': {}, 'T': {}, 'M': {}}
        
        # 更新tau值
        self.update_tau()

        for cid, model in zip(selected_clients, models):
            # 计算准确率提升 (A)
            local_acc = self.clients[cid].test(model)['accuracy']
            
            # 计算客户端模型对全局模型的相对准确率提升
            # A_i = max(0, (local_acc - current_global_acc) / (1 - current_global_acc))
            A_i = 0.5 * local_acc +0.5 * current_global_acc

            # 解释：
            # 1. 这个公式的设计目的是为了更好地体现高准确率区间的提升价值。
            # 2. 除以 (1 - current_global_acc) 使得在接近完美准确率时，相同的绝对提升会得到更高的分数。
            # 3. 这种方法反映了这样一个事实：当全局模型已经很好时，进一步的改进变得更加困难和有价值。
            # 4. 使用 max(0, ...) 确保只有正面贡献被考虑。
            
            raw_scores['A'][cid] = A_i

            # 计算梯度质量 (G)
            grad = fmodule._model_sub(model, self.model)
            grad_norm = fmodule._model_norm(grad).item()
            self.grad_norm_history.append(grad_norm)
            global_grad = fmodule._model_sub(self.model, self.last_global_model) if hasattr(self, 'last_global_model') else grad
            cosine_similarity = self.calculate_cosine_similarity(grad, global_grad)
            G_i = ((1 + cosine_similarity) * min(grad_norm, self.tau)) / self.tau
            raw_scores['G'][cid] = G_i

            # 计算时间衰减因子 (T)
            T_i = np.exp(-self.gamma * (self.current_round - self.last_participation[cid]))
            raw_scores['T'][cid] = T_i

            self.last_participation[cid] = self.current_round

        # 计算委员会奖励 (M)
        committee_rewards = self.calculate_committee_rewards(raw_scores)
        for cid in selected_clients:
            raw_scores['M'][cid] = committee_rewards.get(cid, 0)

        # 归一化所有因素
        normalized_scores = self.normalize_factors(raw_scores)

        # 计算最终得分
        final_scores = {}
        for cid in selected_clients:
            final_scores[cid] = (
                self.w_acc * normalized_scores['A'][cid] +
                self.w_grad * normalized_scores['G'][cid] +
                self.w_time * normalized_scores['T'][cid] +
                self.w_committee * normalized_scores['M'][cid]
            )
            # 将得分添加到历史记录中
            self.scores[cid].append(final_scores[cid])

        # 记录客户端信息
        # 记录客户端信息
        for cid in selected_clients:
            # 计算总得分、总轮数和平均分
            total_score = sum(self.scores[cid])
            total_rounds = len(self.scores[cid])
            average_score = total_score / total_rounds if total_rounds > 0 else 0

            self.gv.logger.info(f"Round {self.current_round}, Client {cid}: "
                                f"Acc Improvement={normalized_scores['A'][cid]:.4f}, "
                                f"Gradient Quality={normalized_scores['G'][cid]:.4f}, "
                                f"Time Decay={normalized_scores['T'][cid]:.4f}, "
                                f"Committee Reward={normalized_scores['M'][cid]:.4f}, "
                                f"Final Score={final_scores[cid]:.4f}, "
                                f"Total Score={total_score:.1f}/{total_rounds}, "
                                f"Average Score={average_score:.4f}")
            
            self.gv.logger.write_var_into_output(f"client_{cid}_metrics_round_{self.current_round}", {
                "acc_improvement": float(normalized_scores['A'][cid]),
                "gradient_quality": float(normalized_scores['G'][cid]),
                "time_decay": float(normalized_scores['T'][cid]),
                "committee_reward": float(normalized_scores['M'][cid]),
                "final_score": float(final_scores[cid]),
                "total_score": f"{total_score:.2f}/{total_rounds}",
                "average_score": float(average_score)
            })

        return final_scores

    def normalize_factors(self, raw_scores):
        """归一化各项得分因素"""
        normalized_scores = {}
        for factor in ['A', 'G', 'T', 'M']:
            values = list(raw_scores[factor].values())
            if len(values) > 1:
                min_val = min(values)
                max_val = max(values)
                if max_val > min_val:
                    normalized_scores[factor] = {cid: (val - min_val) / (max_val - min_val) 
                                                for cid, val in raw_scores[factor].items()}
                else:
                    normalized_scores[factor] = {cid: 1.0 for cid in raw_scores[factor]}
            else:
                normalized_scores[factor] = {cid: 1.0 for cid in raw_scores[factor]}
        return normalized_scores

    def calculate_committee_rewards(self, raw_scores):
        """计算委员会奖励"""
        if not self.committee:
            return {cid: 0 for cid in raw_scores['A']}

        # 计算委员会成员的综合得分
        committee_scores = {}
        for cid in self.committee:
            if cid in raw_scores['A']:
                committee_scores[cid] = (
                    self.w_acc * raw_scores['A'][cid] +
                    self.w_grad * raw_scores['G'][cid] +
                    self.w_time * raw_scores['T'][cid]
                )

        # 对委员会成员进行排序
        sorted_committee = sorted(committee_scores.items(), key=lambda x: x[1], reverse=True)
        
        rewards = {}
        R = len(self.committee)
        for i, (cid, _) in enumerate(sorted_committee):
            rewards[cid] = (R - i) / R

        # 非委员会成员的奖励为0
        for cid in raw_scores['A']:
            if cid not in rewards:
                rewards[cid] = 0

        return rewards

    def update_tau(self):
        """更新tau值"""
        if len(self.grad_norm_history) > 100:  # 确保有足够的历史数据
            self.tau = np.percentile(self.grad_norm_history[-100:], self.tau_percentile)

    def iterate(self):
        """执行每一轮的训练过程"""
        self.gv.logger.info(f"Round {self.current_round} started")
        self.gv.logger.write_var_into_output(f"round_start", self.current_round)

        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        models = res['model']

        current_global_acc = self.test(self.model)['accuracy']
        round_scores = self.calculate_client_scores(self.selected_clients, models, current_global_acc)

        self.last_global_model = self.model  # 保存当前全局模型
        self.model = self.aggregate(models, round_scores)
        return True

    def aggregate(self, models, round_scores):
        """聚合模型"""
        total_score = sum(round_scores.values())
        weights = [round_scores[cid] / total_score for cid in self.selected_clients]

        for cid, weight in zip(self.selected_clients, weights):
            self.gv.logger.info(f"Round {self.current_round}, Client {cid}: Aggregation Weight={weight:.4f}")
            self.gv.logger.write_var_into_output(f"client_{cid}_aggregation_weight_round_{self.current_round}", float(weight))

        return fmodule._model_average(models, weights)

class Client(fedbase.BasicClient):
    def train(self, model):
        """执行本地训练"""
        return super().train(model)

class V2:
    Server = Server
    Client = Client