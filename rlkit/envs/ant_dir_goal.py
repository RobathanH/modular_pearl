import numpy as np
import itertools

from rlkit.envs.ant import AntEnv
from . import register_env

# TODO: Make ctrl-cost, survive-reward and termination condition the same
# between each task type

@register_env('ant-dir-goal')
class AntDirGoalEnv(AntEnv):
    def __init__(self, task={}, n_tasks=4, forward_backward=False, contact_force_in_obs=True, unify_reward_structure=False, randomize_tasks=True, **kwargs):
        assert randomize_tasks, "Non-random tasks are not implemented"
        
        self.forward_backward = forward_backward
        self.contact_force_in_obs = contact_force_in_obs
        self.unify_reward_structure = unify_reward_structure
        
        self.tasks = self.sample_tasks(n_tasks)
        
        if len(task):
            self._task = self._goal = task
        else:
            self._task = self._goal = self.tasks[0]
        
        super().__init__(**kwargs)
        
    def step(self, action):
        if self.unify_reward_structure:
            return self.unified_step(action)
        else:
            if 'dir' in self._task:                
                return self.orig_dir_step(action)
            elif 'goal' in self._task:
                return self.orig_goal_step(action)
            else:
                raise NotImplementedError
        
    def orig_dir_step(self, action):
        direct = (np.cos(self._task['dir']), np.sin(self._task['dir']))
            
        torso_xyz_before = np.array(self.get_body_com("torso"))
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                and state[2] >= 0.2 and state[2] <= 1.0
        terminated = not notdone
        truncated = False
        ob = self._get_obs()
        return ob, reward, terminated, truncated, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )
        
    def orig_goal_step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._task['goal'])) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        terminated = False
        truncated = False
        ob = self._get_obs()
        return ob, reward, terminated, truncated, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )
        
    def unified_step(self, action):
        x_before = np.array(self.get_body_com("torso"))[:2]
        self.do_simulation(action, self.frame_skip)
        x_after = np.array(self.get_body_com("torso"))[:2]
        
        if 'dir' in self._task:
            goal_dir = np.array([np.cos(self._task['dir']), np.sin(self._task['dir'])])
            xvel = (x_after - x_before) / self.dt
            xvel_norm = np.linalg.norm(xvel)
            if xvel_norm == 0:
                main_reward = -10
            else:
                main_reward = np.dot(xvel, goal_dir) - 10
        elif 'goal' in self._task:
            goal_pos = self._task['goal']
            main_reward = -np.sum(np.abs(x_after - goal_pos))
        else:
            raise NotImplementedError
        
        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
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
        self.task_types = {
            "goal": [2 * i for i in range(len(dir_tasks))] + [2 * len(dir_tasks) + i for i in range(len(goal_tasks) - len(dir_tasks))],
            "dir": [2 * i + 1 for i in range(len(dir_tasks))]
        }
        
        return tasks
        
    def _get_obs(self):
        if self.contact_force_in_obs:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat
            ])
        
    def get_all_task_idx(self):
        return range(len(self.tasks))
    
    def reset_task(self, idx):
        self._task = self._goal = self.tasks[idx]
        self.reset()