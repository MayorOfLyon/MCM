import numpy as np
import matplotlib.pyplot as plt

class VirusCellularAutomaton:
    def __init__(self, size, initial_infected, recovery_time=5, infection_probability=0.3):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # 0: Healthy, 1: Infected, 2: Recovered
        self.recovery_time = recovery_time
        self.infection_probability = infection_probability
        self.infection_countdown = np.zeros((size, size), dtype=int)

        # 初始化感染者
        infected_cells = np.random.choice(size * size, initial_infected, replace=False)
        self.grid.ravel()[infected_cells] = 1
        self.infection_countdown.ravel()[infected_cells] = recovery_time

    def apply_rules(self):
        new_grid = self.grid.copy()
        new_countdown = self.infection_countdown.copy()

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 1:  # 如果是感染者
                    if self.infection_countdown[i, j] == 0:
                        new_grid[i, j] = 2  # 感染者康复
                    else:
                        new_countdown[i, j] -= 1

                if self.grid[i, j] == 0:  # 如果是健康者
                    # 计算邻居中感染者的数量
                    infected_neighbors = np.sum(self.grid[max(0, i-1):min(self.size, i+2),
                                                           max(0, j-1):min(self.size, j+2)] == 1)

                    # 根据概率确定是否感染
                    if np.random.rand() < self.infection_probability * infected_neighbors:
                        new_grid[i, j] = 1  # 健康者感染
                        new_countdown[i, j] = self.recovery_time

        self.grid = new_grid
        self.infection_countdown = new_countdown

    def simulate(self, steps):
        for _ in range(steps):
            self.apply_rules()

    def plot(self):
        plt.imshow(self.grid, cmap='seismic', interpolation='nearest', vmin=0, vmax=2)
        plt.colorbar(ticks=[0, 1, 2], label='0: Healthy, 1: Infected, 2: Recovered')
        plt.title('Virus Spread Simulation')
        plt.show()

# 示例：模拟病毒传播
size = 50
initial_infected = 5
simulation_steps = 15

ca = VirusCellularAutomaton(size, initial_infected)
ca.simulate(simulation_steps)
ca.plot()
