import flgo.experiment.analyzer as fea
import matplotlib.pyplot as plt


class AnalyzerAndPainter:
    def analyze_and_plot(self,task, option):
        # 加载所有记录
        records = fea.load_records(task, ['MyAlgorithm'], option)

        # 初始化存储数据的结构
        rounds = []
        local_acc = {}
        global_acc = {}
        grad_norm = {}
        time_decay = {}
        score = {}
        agg_weight = {}
        committee_counts = []
        normal_client_counts = []

        for record in records:
            record_rounds = record.log['round_start']
            rounds.extend(record_rounds)
            
            clients = [f'client_{i}_metrics_round_' for i in range(30)]
            
            for client in clients:
                if client not in local_acc:
                    local_acc[client] = []
                    global_acc[client] = []
                    grad_norm[client] = []
                    time_decay[client] = []
                    score[client] = []
                    agg_weight[client] = []

            for r in record_rounds:
                # 委员会和普通客户端数量
                committee_counts.append(len(record.log[f'committee_round_{r}']))
                normal_client_counts.append(len(record.log[f'normal_clients_round_{r}']))
                
                for client in clients:
                    if f'{client}{r}' in record.log:
                        local_acc[client].append(record.log[f'{client}{r}'][0]['local_acc'])
                        global_acc[client].append(record.log[f'{client}{r}'][0]['global_acc'])
                        grad_norm[client].append(record.log[f'{client}{r}'][0]['grad_norm'])
                        time_decay[client].append(record.log[f'{client}{r}'][0]['time_decay'])
                        score[client].append(record.log[f'{client}{r}'][0]['score'])
                        
                        agg_key = f'{client}_aggregation_weight_round_{r}'
                        if agg_key in record.log:
                            agg_weight[client].append(record.log[agg_key][0])
                        else:
                            agg_weight[client].append(0)
                    else:
                        local_acc[client].append(None)
                        global_acc[client].append(None)
                        grad_norm[client].append(None)
                        time_decay[client].append(None)
                        score[client].append(None)
                        agg_weight[client].append(None)

        return rounds, local_acc, global_acc, grad_norm, time_decay, score, agg_weight, committee_counts, normal_client_counts

    def plot_accuracy(self, rounds, local_acc, global_acc, clients):
        plt.figure(figsize=(10, 6))

        for client in clients[:5]:  # 选择前5个客户端
            plt.plot(rounds, local_acc[client], label=f'{client} Local Acc')
            plt.plot(rounds, global_acc[client], '--', label=f'{client} Global Acc')

        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('客户端模型准确率和全局模型准确率')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_grad_time_decay(self, rounds, grad_norm, time_decay, clients):
        plt.figure(figsize=(10, 6))

        for client in clients[:5]:  # 选择前5个客户端
            plt.plot(rounds, grad_norm[client], label=f'{client} Grad Norm')
            plt.plot(rounds, time_decay[client], '--', label=f'{client} Time Decay')

        plt.xlabel('Round')
        plt.ylabel('Value')
        plt.title('梯度范数和时间衰减因子')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_score_agg_weight(self, rounds, score, agg_weight, clients):
        plt.figure(figsize=(10, 6))

        for client in clients[:5]:  # 选择前5个客户端
            plt.plot(rounds, score[client], label=f'{client} Score')
            plt.plot(rounds, agg_weight[client], '--', label=f'{client} Aggregation Weight')

        plt.xlabel('Round')
        plt.ylabel('Value')
        plt.title('客户端得分和聚合权重')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_committee_normal_clients(self, rounds, committee_counts, normal_client_counts):
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, committee_counts, label='Committee Members')
        plt.plot(rounds, normal_client_counts, '--', label='Normal Clients')

        plt.xlabel('Round')
        plt.ylabel('Number of Clients')
        plt.title('参与客户端数量和委员会成员变化')
        plt.legend()
        plt.grid(True)
        plt.show()
    

