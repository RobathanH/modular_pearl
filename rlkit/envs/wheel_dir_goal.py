import numpy as np
import itertools
import os

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.utils.save_video import save_video

from . import register_env

@register_env('wheel-dir-goal')
class WheelDirGoalEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    
    def __init__(self, task={}, n_tasks=4, forward_backward=False, reward_variant=0, randomize_tasks=True, **kwargs):
        model_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "assets", "wheeled.xml"
        )
        dummy_obs_space = Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self, model_file, 5, observation_space=dummy_obs_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)
        
        # Fix observation space
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=self._get_obs().shape
        )
        
        
        # Set up tasks
        assert randomize_tasks, "Non-random tasks are not implemented"
        self.forward_backward = forward_backward
        self.reward_variant = reward_variant
        
        self.tasks, self.task_types = self.sample_tasks(n_tasks)
        self.task_idx_to_type = {idx: task_type for task_type, indices in self.task_types.items() for idx in indices}
        
        if len(task):
            self._task = self._goal = task
        else:
            self._task = self._goal = self.tasks[0]
            
            
        # Rendering Util
        self.render_mode = "rgb_array"
        self.render_on = False
        self.render_frames = []
        
        
        
    def step(self, action):
        x_before = np.array(self.get_body_com("car"))[:2]
        self.do_simulation(action, self.frame_skip)
        x_after = np.array(self.get_body_com("car"))[:2]
        
        if self.render_on:
            self.render_frames.append(self.render())
            
        if self.reward_variant == 0:
            if 'dir' in self._task:
                goal_dir = np.array([np.cos(self._task['dir']), np.sin(self._task['dir'])])
                xvel = (x_after - x_before) / self.dt
                main_reward = np.dot(xvel, goal_dir)
            elif 'goal' in self._task:
                goal_pos = self._task['goal']
                main_reward = -np.sum(np.abs(x_after - goal_pos))
            else:
                raise NotImplementedError
            
        elif self.reward_variant == 1:
            if 'dir' in self._task:
                goal_dir = np.array([np.cos(self._task['dir']), np.sin(self._task['dir'])])
                xvel = (x_after - x_before) / self.dt
                main_reward = np.dot(xvel, goal_dir)
            elif 'goal' in self._task:
                goal_pos = self._task['goal']
                prev_goal_dist = np.linalg.norm(goal_pos - x_before)
                curr_goal_dist = np.linalg.norm(goal_pos - x_after)
                vel_to_goal = (prev_goal_dist - curr_goal_dist) / self.dt
                main_reward = vel_to_goal
            else:
                raise NotImplementedError
            
        elif self.reward_variant == 2:
            if 'dir' in self._task:
                goal_dir = np.array([np.cos(self._task['dir']), np.sin(self._task['dir'])])
                xvel = (x_after - x_before) / self.dt
                goal_vel = np.dot(xvel, goal_dir)
                main_reward = -np.abs(goal_vel - 10)
            elif 'goal' in self._task:
                goal_pos = self._task['goal']
                main_reward = -np.sum(np.abs(x_after - goal_pos))
            else:
                raise NotImplementedError
            
        else:
            raise NotImplementedError
        
        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = main_reward - ctrl_cost - contact_cost + survive_reward
        
        terminated = False
        truncated = False
        ob = self._get_obs()
        return ob, reward, terminated, truncated, dict(
            reward_main=main_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward
        )
    
    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:-5],
                self.data.qvel.flat[:-5],
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        
        # Set goal position
        if 'dir' in self._task:
            # Dir marker
            angle = self._task['dir']
            qpos[-5:-2] = np.array([0, 0, angle])
            qvel[-5:-2] = 0
            
            # Hide goal marker
            qpos[-2:] = 100
            qvel[-2:] = 0
        elif 'goal' in self._task:
            # Hide dir marker
            qpos[-5:-2] = 100
            qvel[-5:-2] = 0
            
            # Goal marker
            qpos[-2:] = self._task['goal']
            qvel[-2:] = 0
        else:
            raise NotImplementedError
        
        self.set_state(qpos, qvel)
        return self._get_obs()
        
        
        
    def sample_tasks(self, num_tasks):
        num_dir_tasks = num_tasks // 2
        if self.forward_backward:
            num_dir_tasks -= (num_dir_tasks % 2)
            dir_tasks = [{'dir': angle} for angle in [0, np.pi] * (num_dir_tasks // 2)]
        else:
            dir_tasks = [{'dir': angle} for angle in np.random.uniform(0, 2 * np.pi, size=num_dir_tasks)]
        
        num_goal_tasks = num_tasks - num_dir_tasks
        a = np.random.random(num_goal_tasks) * 2 * np.pi
        r = 3 * np.random.random(num_goal_tasks) ** 0.5
        goal_tasks = [{'goal': goal} for goal in np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)]
        
        # Interleave each type of task
        tasks = list(itertools.chain(*zip(goal_tasks[:len(dir_tasks)], dir_tasks))) + goal_tasks[len(dir_tasks):]
        
        # Save which task types have which train and test attributes
        task_types = {
            "goal": [2 * i for i in range(len(dir_tasks))] + [2 * len(dir_tasks) + i for i in range(len(goal_tasks) - len(dir_tasks))],
            "dir": [2 * i + 1 for i in range(len(dir_tasks))]
        }
        
        return tasks, task_types
        
    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self._goal = self.tasks[idx]
        self.reset()
        
        
    def reset(self):
        out = super().reset()
        if self.render_on:
            self.render_frames.append(self.render())
        return out
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent
    
    def begin_render(self):
        self.render_on = True
        self.render_frames = []
        
    def save_render(self, video_folder: str, file_prefix: str = "", episode_index: int = 0):
        if not self.render_on:
            return
        
        save_video(
            self.render_frames,
            video_folder,
            name_prefix=file_prefix,
            episode_index=episode_index,
            fps=self.metadata["render_fps"],
            episode_trigger=lambda i: True # Otherwise it uses bicubic schedule
        )
        
        self.render_on = False
        self.render_frames = []