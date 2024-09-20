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

class ActivityHeterogeneitySimulator_basic(flgo.simulator.base.BasicSimulator):
    def update_client_availability(self):
        if self.gv.clock.current_time==0:
            self.set_variable(self.all_clients, 'prob_available', [1 for _ in self.clients])
            self.set_variable(self.all_clients, 'prob_unavailable', [int(random.random() >= 0.5) for _ in self.clients])
            return
        pa = [0.1 for _ in self.clients]
        pua = [0.1 for _ in self.clients]
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)