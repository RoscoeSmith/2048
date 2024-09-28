from random import random, randint, choice, choices, sample
from collections import namedtuple as ntuple
from copy import deepcopy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SimpleGame:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.grid = [0 for _ in range(rows * cols)]
        self.new_tile_probs = [0.5, 0.5]
        self.score = 0
        self.game_over = False

        self.tile_map: list[str] = [
            '\x1b[100m  \x1b[0m',    # 0
            '\x1b[0;41m1 \x1b[0m',  # 1
            '\x1b[0;42m2 \x1b[0m',  # 2
            '\x1b[0;43m3 \x1b[0m',  # 3
            '\x1b[0;44m4 \x1b[0m',  # 4
            '\x1b[0;45m5 \x1b[0m',  # 5
            '\x1b[0;46m6 \x1b[0m',  # 6
            '\x1b[0;47m7 \x1b[0m',  # 7
        ]

        self.add_new_tile()

    def get_row(self, idx: int) -> list[int]:
        return self.grid[self.cols * idx : self.cols * idx + self.cols]

    def get_col(self, idx: int) -> list[int]:
        return self.grid[idx : self.rows * self.cols : self.cols]

    def set_row(self, idx: int, row: list[int]) -> None:
        self.grid[self.cols * idx : self.cols * idx + self.cols] = row

    def set_col(self, idx: int, col: list[int]) -> None:
        self.grid[idx : self.rows * self.cols : self.cols] = col

    def slide_left(self, items: list[int], save_combined: bool = True) -> list[int]:
        combined = []
        new_tiles = set()
        for source in range(1, len(items)):
            if items[source] == 0:
                continue
            
            bound = -1
            i = source - 1
            while i > bound:
                if items[i] == items[source] and i not in new_tiles:
                    items[i] += 1
                    new_tiles.add(i)
                    combined.append(items[i])
                    break
                elif items[i] == 0:
                    i -= 1
                else:
                    bound = i
            else:
                items[i + 1] = items[source]
                
            if bound + 1 != source:
                items[source] = 0

        if save_combined:
            for n in combined:
                self.score += 2 ** n

        return items

    def slide_right(self, items: list[int]) -> list[int]:
        return self.slide_left(items[::-1])[::-1]

    def slide_row_left(self, idx: int) -> None:
        self.set_row(idx, self.slide_left(self.get_row(idx)))
    
    def slide_row_right(self, idx: int) -> None:
        self.set_row(idx, self.slide_right(self.get_row(idx)))
            
    def slide_col_up(self, idx: int) -> None:
        self.set_col(idx, self.slide_left(self.get_col(idx)))
    
    def slide_col_down(self, idx: int) -> None:
        self.set_col(idx, self.slide_right(self.get_col(idx)))
            
    def slide_grid_left(self) -> None:
        for r in range(self.rows):
            self.slide_row_left(r)

    def slide_grid_right(self) -> None:
        for r in range(self.rows):
            self.slide_row_right(r)

    def slide_grid_up(self) -> None:
        for c in range(self.cols):
            self.slide_col_up(c)
    
    def slide_grid_down(self) -> None:
        for c in range(self.cols):
            self.slide_col_down(c)
    
    def draw_grid(self):
        for r in range(self.rows):
            s = ''
            for i in self.get_row(r):
                try:
                    s += self.tile_map[i]
                except IndexError:
                    s += f'{str(i):<2}'
            yield s

    def print_grid(self) -> None:
        for row in self.draw_grid():
            print(row)

    def print_board(self) -> None:
        self.print_grid()
        print(f"Score: {self.score}")

    def add_new_tile(self) -> None:
        empty_tiles = [i for i in range(self.rows * self.cols) if self.grid[i] == 0]
        if len(empty_tiles) == 0:
            self.game_over = True
        tile_idx = choice(empty_tiles)
        tile_value = choices(list(range(1, len(self.new_tile_probs) + 1)), self.new_tile_probs, k=1)[0]
        self.grid[tile_idx] = tile_value

    def do_action(self, dir: str) -> bool:
        old_grid = self.grid.copy()
        match dir:
            case 'left':
                self.slide_grid_left()
            case 'right':
                self.slide_grid_right()
            case 'up':
                self.slide_grid_up()
            case 'down':
                self.slide_grid_down()
            case _:
                return
        if old_grid != self.grid:
            self.add_new_tile()
            rval = True
        else:
            rval = False

        if not self.has_legal_move():
            self.game_over = True

        return rval

    def has_legal_move(self) -> None:
        # Check for empty tile
        if 0 in self.grid:
            return True
        
        # Check rows
        for r in range(self.rows):
            row = self.get_row(r)
            for i in range(1, len(row)):
                if row[i] == row[i - 1]:
                    return True
                
        # Check cols
        for c in range(self.cols):
            col = self.get_col(c)
            for i in range(1, len(col)):
                if col[i] == col[i - 1]:
                    return True

        return False


class Agent:
    def __init__(self, game: SimpleGame, *args) -> None:
        self.game = game
        self.cinit(args)

    def cinit(self, args):
        pass

class HumanAgent(Agent):    
    def get_action(self) -> None:
        match input(">").lower():
            case 'w':
                dir = 'up'
            case 's':
                dir = 'down'
            case 'a':
                dir = 'left'
            case 'd':
                dir = 'right'
            case _:
                return
        self.game.do_action(dir)            

class NNAgent(Agent):
    ntuple('SARS', ('state', 'action', 'reward', 'next_state'))
    def cinit(self, args):
        self.buffer: list[SARS] = []

    def get_action(self, dir: int) -> bool:
        match dir:
            case 0:
                d = 'down'
            case 1:
                d = 'left'
            case 2:
                d = 'up'
            case 3:
                d = 'right'
            case _:
                return
        return self.game.do_action(d)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dims = (16, 128, 128, 4)
        self.fc1 = nn.Linear(self.dims[0], self.dims[1])
        self.fc2 = nn.Linear(self.dims[1], self.dims[2])
        self.fc3 = nn.Linear(self.dims[2], self.dims[3])


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        



def load_model(path):
    model = torch.load(path)
    agent = NNAgent(SimpleGame(4, 4))
    while not agent.game.game_over:
        agent.game.print_board()
        actions = torch.argsort(model(torch.tensor(agent.game.grid, dtype = torch.float32)), descending = True)
        for action in actions:
            if agent.get_action(action):
                break
        # input()
    agent.game.print_board()
    print("Game Over")

def train_model(path):
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.005
    batch_size = 64
    target_update_frequency = 10

    policy_net = Net()
    target_net = Net()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    replay_buffer = []
    losses = []

    def play_step(state, epsilon):
        #e-greedy action selection
        if random() < epsilon:
            action = randint(0, 3)
        else:
            with torch.no_grad():
                q_values = policy_net(torch.tensor(state, dtype = torch.float32))
                action = torch.argmax(q_values).item()  # Greedy
        return action

    def update_network():
        if len(replay_buffer) < batch_size:
            return 0

        # Sample random minibatch
        minibatch = sample(replay_buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)

        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        # Current Q-values for chosen actions
        current_q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

        # Target Q-values
        with torch.no_grad():
            next_q_values = target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

        # Compute loss and optimize the model
        loss = criterion(current_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    num_episodes = 5000
    for episode in range(num_episodes):
        agent = NNAgent(SimpleGame(4, 4))
        state = agent.game
        done = False
        total_reward = 0
        episode_loss = 0

        # print(f"Starting episode {episode}")

        while not done:
            # print("Current game state:")
            # state.print_board()

            agent.game = deepcopy(state)
            action = play_step(state.grid, epsilon)
            changed = agent.get_action(action)
            next_state = agent.game
            reward = next_state.score - state.score
            if changed and reward == 0:
                reward += 1
            if next_state.game_over:
                done = True
                reward = -1024
            total_reward += reward

            # print(f"Action: {action}, reward: {reward}")
            # input()

            replay_buffer.append((state.grid, action, reward, next_state.grid, done))
            if len(replay_buffer) > 10000:
                replay_buffer.pop(0)

            episode_loss += update_network()

            state = next_state

        losses.append(episode_loss)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # input("...")

        print(f'Episode {episode}, Total Reward: {total_reward}')
    
    torch.save(target_net, path)

    # Plotting the loss curve
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Curve')
    plt.show()

def play_human():
    print('\x1b[2J')
    g = SimpleGame(4, 4)
    h = HumanAgent(g)
    
    while not g.game_over:
        g.print_board()
        print()
        h.get_action()
        print('\x1b[2J')

    g.print_board()
    print("GAME OVER")
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train_model(sys.argv[2])
        elif sys.argv[1] == 'load':
            load_model(sys.argv[2])
    else:
        play_human()
