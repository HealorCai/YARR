from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from pdb import set_trace
import time
class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False, device = None):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # print(f'obs: {obs}')
            # print(f"rgb: {obs['front_rgb'].shape}")
            # print(f"pcd: {obs['front_point_cloud'].shape}")
            # print(f"left_gripper_pose: {obs['left_gripper_pose'].shape}")
            # print(f"lang_emb: {obs['lang_goal_emb'].shape}")

            # rgb: (3, 256, 256)
            # pcd: (3, 256, 256)
            # left_gripper_pose: (7,)
            # lang_emb: lang_emb: torch.Size([1, 53, 512])
            # set_trace()
        else:
            obs = env.reset()

        # 'task_name': 'bimanual_push_box'
        # 'variation_num': 0


        
        agent.reset()
        for k, v in obs.items():
            if isinstance(v, float):
                obs[k] = np.float32(v)
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items() }
        
        # obs_history = {}
        # non_tensor_obs = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items() if not isinstance(v, torch.Tensor)}
        # tensor_obs = {k: [np.array(v.numpy(), dtype=v.numpy().dtype)] * timesteps for k, v in obs.items() if isinstance(v, torch.Tensor)}
        # obs_history.update(non_tensor_obs)
        # obs_history.update(tensor_obs)

        time_start = time.time()

        for step in range(episode_length):
            print(f'step: {step}')
            print(f'prepped_data_device:{device}')
            prepped_data = {k:torch.tensor(np.array(v)[None], device=device) for k, v in obs_history.items()}
            # prepped_data = {k:torch.tensor(np.array(v)[None], device=self._env_device) for k, v in obs_history.items()}

            # print(f"rgb: {prepped_data['front_rgb'].shape}")
            # print(f"pcd: {prepped_data['front_point_cloud'].shape}")
            # print(f"left_gripper_pose: {prepped_data['left_gripper_pose'].shape}")
            # print(f"lang_emb: {prepped_data['lang_goal_emb'].shape}")
            # set_trace()

            # rgb: torch.Size([1, 1, 3, 256, 256])
            # pcd: torch.Size([1, 1, 3, 256, 256])
            # left_gripper_pose: torch.Size([1, 1, 7])
            # lang_emb: torch.Size([1, 1, 1, 53, 512])

            act_result = agent.act(env._rlbench_env._pyrep, step_signal.value, prepped_data, deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}
            # set_trace()
            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)


            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                time_end = time.time()
                print(f"rollout time: {time_end-time_start:.3f}s")
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    print(f'prepped_data_device:{device}')
                    prepped_data = {k: torch.tensor([v], device=device) for k, v in obs_history.items()}
                    # prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(env._rlbench_env._pyrep, step_signal.value, prepped_data, deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return


