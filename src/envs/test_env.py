from .multiagentenv import MultiAgentEnv
import numpy as np
import random
from PIL import Image, ImageDraw
import os
from pathlib import Path
from PIL import ImageFont


class PursuitEnv(MultiAgentEnv):

    def __init__(self, agent_num=3, catch_threshold=2, map_size=(8, 8)):
        self.n_agents, self.catch_threshold = agent_num, catch_threshold
        self.map_size = map_size
        self.action_num = 5
        self.episode_limit = 100
        
        self.count_steps = 0
        self.record_eps = 0
        self.save_render = False
        self.save_folder = ""

        # init position
        self.agents_pos = np.zeros((self.n_agents, 2), dtype=np.int8)
        self.target_pos = np.zeros(2, dtype=np.int8)

        self._random_position()
    
    def _random_position(self):
        temp_pos = np.zeros((self.n_agents + 1, 2), dtype=np.int8)
        indices = random.sample(range(self.map_size[0] * self.map_size[1]), self.n_agents + 1)
        for i, loc in enumerate(indices):
            temp_pos[i, :] = [loc // self.map_size[1], loc % self.map_size[1]]
        self.agents_pos = temp_pos[:-1, :]
        self.target_pos = temp_pos[-1, :]    
        # print(self.agents_pos, "\n", self.target_pos)

    def _try_step(self, pos, action_id):
        # 1:out of map, 2:crash agent, 3:crash target.
        if action_id == 4:
            return 0, pos
        else:
            step_action = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
            agent_pos_list = [tuple(pos) for pos in self.agents_pos]
            new_pos = pos + step_action[action_id]
            if new_pos[0] < 0 or new_pos[0] >= self.map_size[0] or new_pos[1] < 0 or new_pos[1] >= self.map_size[1]:
                return 1, pos
            elif tuple(new_pos) in agent_pos_list:
                return 2, pos
            elif tuple(new_pos) == tuple(self.target_pos):
                return 3, new_pos
            return 0, new_pos

    def step(self, actions):
        """ Returns reward, terminated, info """
        reward_map = [-0.1, -1, -1, 10]
        done = False
        info = {"Reward": [0] * self.n_agents,
                "Team_Reward": 0}
        self.count_steps += 1

        reward = [0] * self.n_agents
        team_reward = 0
        for i in range(self.n_agents):
            crash, self.agents_pos[i, :] = self._try_step(self.agents_pos[i, :], actions[i])
            reward[i] = reward_map[crash]
            if crash == 3:
                done = True
                team_reward = reward_map[3]
        
        if not done:
            self._target_step(mode="random_escape")

            round_area = self.target_pos + np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int8)
            round_area = [tuple(pos) for pos in round_area]
            count = 0
            for i in range(self.n_agents):
                if tuple(self.agents_pos[i]) in round_area:
                    count += 1
            if count >= self.catch_threshold:
                done = True
                reward = [reward_map[3]] * self.n_agents
                team_reward = reward_map[3]
        
        if self.save_render and self.record_eps % 10000 == 0:
            self.render(save_path=os.path.join(Path(__file__).parents[2], "render", self.save_folder), file_name="{:04d}.png".format(self.count_steps))
        
        if self.count_steps >= self.episode_limit:
            done = True
        
        if done:
            self.record_eps += 1
        
        info["Reward"] = reward
        info["Team_Reward"] = team_reward

        return None, reward, done, info

    def _target_step(self, mode="random"):
        if mode == "random":
            _, self.target_pos = self._try_step(self.target_pos, np.random.randint(0, 5))
        elif mode == "escape": # may stay in corner
            action = 4
            relate_dis = self.agents_pos - self.target_pos 
            near_id = np.argmin(np.sum(np.abs(relate_dis), axis=1))
            if abs(relate_dis[near_id][0]) > abs(relate_dis[near_id][1]) and relate_dis[near_id][0] > 0:
                action = 0
            elif abs(relate_dis[near_id][0]) > abs(relate_dis[near_id][1]) and relate_dis[near_id][0] <= 0:
                action = 1
            elif abs(relate_dis[near_id][0]) <= abs(relate_dis[near_id][1]) and relate_dis[near_id][1] > 0:
                action = 2
            else:
                action = 3
            _, self.target_pos = self._try_step(self.target_pos, action)
        elif mode == "random_escape":
            action = 4
            relate_dis = self.agents_pos - self.target_pos 
            chess_dis = np.sum(np.abs(relate_dis), axis=1)
            if np.min(chess_dis) >= 4:
                action = np.random.randint(0, 5)
            else:
                near_id = np.argmin(chess_dis)
                if abs(relate_dis[near_id][0]) > abs(relate_dis[near_id][1]) and relate_dis[near_id][0] > 0:
                    action = 0
                elif abs(relate_dis[near_id][0]) > abs(relate_dis[near_id][1]) and relate_dis[near_id][0] <= 0:
                    action = 1
                elif abs(relate_dis[near_id][0]) <= abs(relate_dis[near_id][1]) and relate_dis[near_id][1] > 0:
                    action = 2
                else:
                    action = 3
            _, self.target_pos = self._try_step(self.target_pos, action)
        else:
            raise ValueError("ERROR Value of mode.")

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs = np.concatenate((self.agents_pos, [self.target_pos]), axis=0).reshape(-1)
        obs_list = []
        for i in range(self.n_agents):
            obs_list.append(obs)
        return obs_list

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return 2 * (self.n_agents + 1)

    def get_state(self):
        state = np.concatenate((self.agents_pos, [self.target_pos]), axis=0).reshape(-1)
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return 2 * (self.n_agents + 1)

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.action_num))

    def get_total_actions(self):
        return self.action_num

    def reset(self):
        """ Returns initial observations and states"""
        self.count_steps = 0
        self._random_position()

        self.save_folder = "record_{:07d}".format(self.record_eps)

        if self.save_render and self.record_eps % 10000 == 0:
            self.render(save_path=os.path.join(Path(__file__).parents[2], "render", self.save_folder), file_name="{:04d}.png".format(self.count_steps))

    def render(self, save_path="", file_name="render.png"):
        cell_width, cell_height = 100, 100
        radius = 30
        map_h, map_w = self.map_size
        offset = 5

        img_width = cell_width * map_w + 2 * offset
        img_height = cell_height * map_h + 2 * offset

        color = ["#D99694", "#C3D69B"]
        font = ImageFont.truetype("msyh.ttc", 18)
        image = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(image)

        for i in range(map_h + 1):
            draw.line((offset, i * cell_height + offset, img_width - offset, i * cell_height + offset), fill='black')
        for i in range(map_w + 1):
            draw.line((i * cell_width + offset, offset, i * cell_width + offset, img_height - offset), fill='black')

        def draw_circle(pos, c, text=None):
            left_up = (offset + cell_width * (pos[1] + 0.5) - radius,
                    offset + cell_height * (pos[0] + 0.5) - radius)
            right_down = (offset + cell_width * (pos[1] + 0.5) + radius,
                        offset + cell_height * (pos[0] + 0.5) + radius)
            draw.ellipse([left_up, right_down], fill=c)
            if text:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                draw.text(((left_up[0] + right_down[0] - text_width) / 2, (left_up[1] + right_down[1] - text_height) / 2), 
                          text, fill='black', font=font)

        draw_circle(self.target_pos, color[1])

        for i in range(self.n_agents):
            draw_circle(self.agents_pos[i], color[0], str(i + 1))

        # image.show()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            image.save(os.path.join(save_path, file_name))

    def close(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info


if __name__ == "__main__":
    """
    when run this code, change
    from .multiagentenv import MultiAgentEnv
    to
    from multiagentenv import MultiAgentEnv
    """
    test = PursuitEnv()
    test.reset()
    for i in range(100):
        r_action = np.random.randint(0, 5, 3)
        test.step(r_action)
