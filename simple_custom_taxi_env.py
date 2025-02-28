import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output

'''
This is just a simple environment. The specifications for the real testing environment are provided in the spec.
You are free to modify this file to match the real environment and train your own agent. Good luck!
'''
class SimpleTaxiEnv(gym.Wrapper):
    def __init__(self, grid_size=5, fuel_limit=50):
        self.grid_size = grid_size
        env = gym.make("Taxi-v3", render_mode="ansi")
        super().__init__(env)
        
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit

        self.stations = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)]
        self.passenger_loc = None
        self.passenger_picked_up = False  # Track if the passenger has been picked up before

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_fuel = self.fuel_limit

        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(obs)

        taxi_row = min(taxi_row, self.grid_size - 1)
        taxi_col = min(taxi_col, self.grid_size - 1)

        # 🚖 Initialize passenger position
        if pass_idx == 4:  # Passenger is inside the taxi
            self.passenger_loc = (taxi_row, taxi_col)
            self.passenger_picked_up = True
        else:
            self.passenger_loc = self.stations[pass_idx]
            self.passenger_picked_up = False  # Passenger has not been picked up yet

        destination_x, destination_y = self.stations[dest_idx]

        return (taxi_row, taxi_col, *self.passenger_loc, destination_x, destination_y), info

    def step(self, action):
        self.current_fuel -= 1
        obs, reward, terminated, truncated, info = super().step(action)

        if reward == 20:  # Correct `DROPOFF`
            reward = 50
        elif reward == -1:  # Regular movement
            reward = -0.1
        elif reward == -10:  # Incorrect `PICKUP` or `DROPOFF`
            reward = -10

        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(obs)

        taxi_row = min(taxi_row, self.grid_size - 1)
        taxi_col = min(taxi_col, self.grid_size - 1)

        # 🚖 Passenger state management
        if pass_idx == 4:  # Passenger is inside the taxi
            self.passenger_loc = (taxi_row, taxi_col)  # Keep passenger moving with the taxi
            self.passenger_picked_up = True
        elif self.passenger_picked_up:  # Passenger was picked up but is now outside
            self.passenger_loc = (taxi_row, taxi_col)  # Passenger is dropped at taxi's current position
            self.passenger_picked_up = False  # Reset pickup state, allowing another `PICKUP`

        destination_x, destination_y = self.stations[dest_idx]

        return (taxi_row, taxi_col, *self.passenger_loc, destination_x, destination_y), reward, terminated, truncated, info

    def render_env(self, taxi_pos, passenger_pos, destination_pos):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        # Passenger
        px, py = passenger_pos
        if (px, py) != (-1, -1) and 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[px][py] = 'P'

        # Destination
        dx, dy = destination_pos
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dx][dy] = 'D'

        # Taxi
        tx, ty = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[tx][ty] = '🚖'

        # Print map
        for row in grid:
            print(" ".join(row))
        print("\n")

def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        taxi_row, taxi_col, passenger_x, passenger_y, destination_x, destination_y = obs

        if render:
            print(f"Step={step_count}")
            env.render_env((taxi_row, taxi_col), (passenger_x, passenger_y), (destination_x, destination_y))
            time.sleep(0.5)

        action = student_agent.get_action(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "grid_size": 3,  
        "fuel_limit": 10000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")
