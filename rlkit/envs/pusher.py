import numpy as np
import itertools
import os

from gymnasium.envs.mujoco import MuJocoPyEnv, MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.utils.save_video import save_video

from . import register_env

@register_env('pusher')
class PusherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    
    def __init__(self, task={}, n_tasks=3, task_types=None, randomize_tasks=True, **kwargs):
        model_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "assets", "pusher_env.xml"
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
        
        self.valid_task_types = ["to_goal", "from_goal", "to_blocks", "from_blocks"]
        if task_types is not None:
            assert len(set(task_types) - set(self.valid_task_types)) == 0, "Unrecognized task type given"
            self.valid_task_types = task_types
        
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
        self.do_simulation(action, self.frame_skip)
        
        if self.render_on:
            self.render_frames.append(self.render())
        
        block_pos = self.data.qpos.flat[3:-2].reshape(5, 2)
        task_block_ind = self._task["task_block"]
        task_block_pos = block_pos[task_block_ind]
        
        if self._task["type"] in ["to_goal", "from_goal"]:
            goal_pos = self._task["goal"]
            dist_goal = np.linalg.norm(task_block_pos - goal_pos)
            
            if self._task["type"] == "to_goal":
                reward = -1 * dist_goal
            elif self._task["type"] == "from_goal":
                start_dist_goal = np.linalg.norm(self._task["block_pos"][task_block_ind] - goal_pos)
                reward = min(0, dist_goal - 2 * start_dist_goal)
                
        elif self._task["type"] in ["to_blocks", "from_blocks"]:
            dist_blocks = np.linalg.norm(task_block_pos[None, :] - block_pos, axis=1)
            mean_dist_blocks = np.sum(dist_blocks) / (len(block_pos) - 1)
            
            if self._task["type"] == "to_blocks":
                reward = -1 * mean_dist_blocks
            elif self._task["type"] == "from_blocks":
                start_dist_blocks = np.linalg.norm(self._task["block_pos"][task_block_ind][None, :] - self._task["block_pos"], axis=1)
                start_mean_dist_blocks = np.sum(start_dist_blocks) / (len(block_pos) - 1)
                reward = min(0, mean_dist_blocks - 2 * start_mean_dist_blocks)
            
        else:
            raise NotImplementedError
        
        # Add small reward bonus based on distance between pusher and task block
        pusher_pos = self.data.qpos.flat[:2]
        dist_pusher = np.linalg.norm(task_block_pos - pusher_pos)
        reward -= dist_pusher * 0.2
        
        terminated = False
        truncated = False
        ob = self._get_obs()
        return ob, reward, terminated, truncated, dict()
    
    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flat[:-2],
                self.data.qvel.flat[:-2]
            ]
        )
    
    def reset_model(self):
        #self.data.body_xpos[-1][0:2] = self._task["goal"]
        #self.get_body_com("target")[0:2] = self._task["goal"]
        
        qpos = np.concatenate([self._task["pusher_pos"], self._task["block_pos"].flatten(), self._task["goal"]])
        qvel = np.zeros((self.model.nv,))
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    
    
    
    def sample_tasks(self, num_tasks):
        tasks = []
        task_types = {task_type: [] for task_type in self.valid_task_types}
        
        blockarr = np.arange(5)
        for task_ind in range(num_tasks):
            task = {}
            task_type = self.valid_task_types[task_ind % len(self.valid_task_types)]
            task["type"] = task_type
            task["pusher_pos"] = np.zeros((3,))
            
            np.random.shuffle(blockarr)
            
            blockpositions = np.zeros((5, 2))
            for i in range(5):
                xpos = np.random.uniform(.35, .65)
                ypos = np.random.uniform(-.5 + 0.2*i, -.3 + 0.2*i)
                blocknum = blockarr[i]
                blockpositions[blocknum, 0] = -0.2*(blocknum + 1) + xpos
                blockpositions[blocknum, 1] = ypos
            task["block_pos"] = blockpositions
                
            i = np.random.choice(5)
            task["task_block"] = blockarr[i]
            if task_type in ["to_goal", "from_goal"]:
                goal_xpos = np.random.uniform(.75, .95)
                goal_ypos = np.random.uniform(-.5 + .2*i, -.3 + .2*i)
                task["goal"] = np.array([goal_xpos, goal_ypos])
            else:
                task["goal"] = np.zeros((2,))
            
            tasks.append(task)
            task_types[task_type].append(task_ind)
        
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