import os
import time
from datetime import datetime
from collections import defaultdict, deque
from typing import Any, Dict, List
import numpy as np
import pickle

import torch
import tqdm

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import json
from datetime import datetime
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only
import pickle

class FALCONEvaluator(Evaluator):
    """
    Only difference is record the success rate of each episode while evaluating.
    Similar to ORCAEvaluator.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        success_cal = 0 ## my added
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        actions_record = defaultdict(list)
        agent.eval()
        # 初始化评估数据收集（按照训练时的格式）
        eval_data_collection = {
            'observations': {},
            'actions': [],
            'rewards': [],
            'masks': [],
            'info_data': [],
            'running_episode_stats': {},
            'episode_info': {},
            'trajectory': [],
            'total_steps': 0
        }
        
        # 初始化模仿学习数据收集（按照用户要求的格式）
        imitation_learning_data = {
            'jaw_rgb_data': [[] for _ in range(envs.num_envs)],  # 每个环境一个列表
            'jaw_depth_data': [[] for _ in range(envs.num_envs)],  # 每个环境一个列表
            'other_data': {
                'actions': [[] for _ in range(envs.num_envs)],
                'rewards': [[] for _ in range(envs.num_envs)],
                'masks': [[] for _ in range(envs.num_envs)],
                'info_data': [[] for _ in range(envs.num_envs)],
                'pointgoal_with_gps_compass': [[] for _ in range(envs.num_envs)]
            }
        }
        
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # 收集动作数据（按照训练时的格式）
            if self._save_eval_data:
                eval_data_collection['actions'].append(action_data.env_actions.cpu().clone())
                eval_data_collection['total_steps'] += 1
            
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if hasattr(agent, '_agents') and agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )

                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = {
                        "type": "array",
                        "value": action_value.tolist()
                    }
                else:
                    stored_action = {
                        "type": "array",
                        "value": np.array(action_value).tolist()
                    }

                actions_record[episode_key].append(stored_action)
                
                # 收集轨迹数据
                if self._save_eval_data:
                    trajectory_data = {
                        'step': len(eval_data_collection['actions']) - 1,
                        'action': action_value.tolist() if isinstance(action_value, np.ndarray) else action_value,
                        'scene_id': current_episodes_info[i].scene_id,
                        'episode_id': current_episodes_info[i].episode_id
                    }
                    
                    # 添加GPS和Compass信息（如果可用）
                    if 'pointgoal_with_gps_compass' in batch:
                        gps_compass = batch['pointgoal_with_gps_compass'][i]
                        if len(gps_compass) >= 2:
                            trajectory_data['position'] = gps_compass[:2].cpu().numpy().tolist()
                        if len(gps_compass) >= 3:
                            trajectory_data['heading'] = gps_compass[2].item()
                    
                    eval_data_collection['trajectory'].append(trajectory_data)

            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore
            
            # 收集观测数据（按照训练时的格式）
            if self._save_eval_data and self._save_observations:
                for sensor_name, sensor_data in batch.items():
                    if sensor_name not in eval_data_collection['observations']:
                        eval_data_collection['observations'][sensor_name] = []
                    # 保存当前步骤的所有环境的观测数据
                    eval_data_collection['observations'][sensor_name].append(sensor_data.cpu().clone())
            
            # 收集模仿学习数据（按照用户要求的格式）
            if self._save_eval_data:
                for i in range(envs.num_envs):
                    # 收集RGB图像数据
                    if 'agent_0_articulated_agent_jaw_rgb' in batch:
                        rgb_data = batch['agent_0_articulated_agent_jaw_rgb'][i].cpu().numpy()  # (H, W, C)
                        if rgb_data.dtype != np.uint8:
                            rgb_data = (rgb_data * 255).astype(np.uint8)
                        imitation_learning_data['jaw_rgb_data'][i].append(rgb_data)
                        # print(f"[DEBUG] 收集 RGB 数据: env={i}, shape={rgb_data.shape}, dtype={rgb_data.dtype}")
                    else:
                        print(f"[DEBUG] 未找到 'agent_0_articulated_agent_jaw_rgb' 键，batch 中的键: {list(batch.keys())}")
                    
                    # 收集深度图像数据
                    if 'agent_0_articulated_agent_jaw_depth' in batch:
                        depth_data = batch['agent_0_articulated_agent_jaw_depth'][i].cpu().numpy()  # (H, W) or (H, W, 1)
                        if depth_data.ndim == 2:
                            depth_data = np.expand_dims(depth_data, axis=-1)  # 添加最后一维
                        if depth_data.dtype != np.float32:
                            depth_data = depth_data.astype(np.float32)
                        imitation_learning_data['jaw_depth_data'][i].append(depth_data)
                        # print(f"[DEBUG] 收集深度数据: env={i}, shape={depth_data.shape}, dtype={depth_data.dtype}")
                    else:
                        print(f"[DEBUG] 未找到 'agent_0_articulated_agent_jaw_depth' 键，batch 中的键: {list(batch.keys())}")
                    
                    # 收集GPS/Compass数据
                    if 'pointgoal_with_gps_compass' in batch:
                        gps_compass = batch['pointgoal_with_gps_compass'][i].cpu().numpy().astype(np.float32)
                        if len(gps_compass) >= 2:
                            imitation_learning_data['other_data']['pointgoal_with_gps_compass'][i].append(gps_compass[:2])
                        else:
                            imitation_learning_data['other_data']['pointgoal_with_gps_compass'][i].append(gps_compass)
                    
                    # 收集动作数据（从之前保存的动作中获取）
                    if len(eval_data_collection['actions']) > 0:
                        action_data = eval_data_collection['actions'][-1][i].cpu().numpy()
                        if action_data.ndim == 0:
                            action_data = np.array([action_data], dtype=np.int64)
                        elif action_data.dtype != np.int64:
                            action_data = action_data.astype(np.int64)
                        imitation_learning_data['other_data']['actions'][i].append(action_data)
                    
                    # 收集奖励数据
                    reward_value = rewards_l[i] if i < len(rewards_l) else 0.0
                    imitation_learning_data['other_data']['rewards'][i].append(np.float64(reward_value))
                    
                    # 收集mask数据
                    mask_value = not dones[i] if i < len(dones) else True
                    imitation_learning_data['other_data']['masks'][i].append(np.float64(mask_value))
                    
                    # 收集info数据
                    info_copy = infos[i].copy() if i < len(infos) else {}
                    imitation_learning_data['other_data']['info_data'][i].append(info_copy)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        print(f"Till now Success Rate: {success_cal/(len(stats_episodes)+1)}")
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    
                    # 收集episode信息（按照训练时的格式）
                    if self._save_eval_data and self._save_episode_info:
                        episode_key = f"{current_episodes_info[i].scene_id}_{current_episodes_info[i].episode_id}_{ep_eval_count.get((current_episodes_info[i].scene_id, current_episodes_info[i].episode_id), 0) + 1}"
                        eval_data_collection['episode_info'][episode_key] = {
                            'total_reward': current_episode_reward[i].item(),
                            'final_reward': rewards_l[i],
                            'episode_length': len(eval_data_collection['actions']),
                            'scene_id': current_episodes_info[i].scene_id,
                            'episode_id': current_episodes_info[i].episode_id,
                            'episode_stats': episode_stats.copy(),
                            'info_data': infos[i].copy(),
                            'success': disp_info.get('success', False),
                            'spl': disp_info.get('spl', 0.0),
                            'distance_to_goal': disp_info.get('distance_to_goal', 0.0)
                        }
                    
                    # 收集奖励数据
                    if self._save_eval_data and self._save_rewards:
                        reward_key = f"{current_episodes_info[i].scene_id}_{current_episodes_info[i].episode_id}_{ep_eval_count.get((current_episodes_info[i].scene_id, current_episodes_info[i].episode_id), 0) + 1}"
                        eval_data_collection['rewards'].append({
                            'episode_key': reward_key,
                            'total_reward': current_episode_reward[i].item(),
                            'final_reward': rewards_l[i],
                            'step_rewards': rewards_l  # 当前步骤的奖励
                        })
                    
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        # show scene and episode
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        print(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.") # for debug
                        
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )
                    
                    # 保存模仿学习数据（按照用户要求的格式）
                    print(f"[DEBUG] Episode 结束，检查保存条件: _save_eval_data = {self._save_eval_data}")
                    if self._save_eval_data:
                        print(f"[DEBUG] 开始保存模仿学习数据，环境 {i}")
                        self._save_imitation_learning_data(
                            imitation_learning_data, 
                            current_episodes_info[i], 
                            episode_stats, 
                            i, 
                            checkpoint_index
                        )
                    else:
                        print(f"[DEBUG] 跳过保存，_save_eval_data = {self._save_eval_data}")
                        
                        # 清空当前环境的数据，为下一个episode做准备
                        imitation_learning_data['jaw_rgb_data'][i] = []
                        imitation_learning_data['jaw_depth_data'][i] = []
                        for key in imitation_learning_data['other_data']:
                            imitation_learning_data['other_data'][key][i] = []

            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # ==== 保存 result.json ====
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
                            "SR": round(aggregated_stats.get("success", 0), 4),
                            "SPL": round(aggregated_stats.get("spl", 0), 4),
                            "PSC": round(aggregated_stats.get("psc", 0), 4),
                            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
                            "Total": round(
                                0.4 * aggregated_stats.get("success", 0)
                                + 0.3 * aggregated_stats.get("spl", 0)
                                + 0.3 * aggregated_stats.get("psc", 0),
                                4,
                                    ),
                        }

        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)

        # ==== 保存 actions.json ====
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)
            
        # ==== 保存评估数据（按照训练时的格式）====
        if self._save_eval_data:
            self._save_evaluation_data(eval_data_collection, checkpoint_index)
    
    def _save_evaluation_data(self, eval_data_collection, checkpoint_index):
        """保存评估数据，格式与训练时完全一致"""
        # 准备保存的数据，完全按照训练时的格式
        imitation_data = {
            'observations': {},
            'actions': None,
            'rewards': None,
            'masks': None,
            'info_data': None,
            'running_episode_stats': None,
            'update_count': checkpoint_index,
            'num_steps': eval_data_collection['total_steps'],
            'num_envs': self.envs.num_envs if hasattr(self, 'envs') else 1
        }
        
        # 保存观测数据，格式与训练时一致
        if self._save_observations and eval_data_collection['observations']:
            for sensor_name, sensor_data_list in eval_data_collection['observations'].items():
                if sensor_data_list:
                    # 将列表中的tensor堆叠成一个tensor，格式: (num_steps, num_envs, ...)
                    stacked_data = torch.stack(sensor_data_list, dim=0).cpu().clone()
                    imitation_data['observations'][sensor_name] = stacked_data
                    print(f"[DEBUG] 保存传感器 {sensor_name} 数据，形状: {stacked_data.shape}")
        
        # 保存动作数据，格式与训练时一致
        if eval_data_collection['actions']:
            # 将动作列表堆叠成tensor，格式: (num_steps, num_envs, action_dim)
            actions_tensor = torch.stack(eval_data_collection['actions'], dim=0).cpu().clone()
            imitation_data['actions'] = actions_tensor
            print(f"[DEBUG] 保存动作数据，形状: {actions_tensor.shape}")
        
        # 保存奖励数据，格式与训练时一致
        if self._save_rewards and eval_data_collection['rewards']:
            # 收集所有episode的奖励数据
            all_rewards = []
            for episode_reward_info in eval_data_collection['rewards']:
                total_reward = episode_reward_info.get('total_reward', 0.0)
                all_rewards.append(total_reward)
            
            if all_rewards:
                # 转换为tensor格式: (num_episodes, 1)
                rewards_tensor = torch.tensor(all_rewards).unsqueeze(-1)
                imitation_data['rewards'] = rewards_tensor
                print(f"[DEBUG] 保存奖励数据，形状: {rewards_tensor.shape}")
        
        # 保存episode信息作为info_data
        if self._save_episode_info and eval_data_collection['episode_info']:
            # 将episode信息转换为与训练时一致的格式
            info_data = {}
            for key, value in eval_data_collection['episode_info'].items():
                if isinstance(value, list):
                    info_data[key] = torch.tensor(value)
                else:
                    info_data[key] = torch.tensor([value])
                print(f"[DEBUG] 保存info数据 {key}，形状: {info_data[key].shape}")
            imitation_data['info_data'] = info_data
        
        # 保存轨迹数据作为running_episode_stats的一部分
        if eval_data_collection['trajectory']:
            running_stats = {}
            # 将轨迹数据转换为tensor格式
            trajectory_data = eval_data_collection['trajectory']
            if isinstance(trajectory_data, list) and len(trajectory_data) > 0:
                # 如果轨迹数据是列表，尝试转换为tensor
                try:
                    running_stats['trajectory'] = torch.tensor(trajectory_data)
                except:
                    # 如果无法直接转换，保存为原始格式
                    running_stats['trajectory_raw'] = trajectory_data
            imitation_data['running_episode_stats'] = running_stats
            print(f"[DEBUG] 保存轨迹数据作为running_episode_stats")
        
        # 为每次评估创建独立的文件夹，与训练时的格式一致
        update_folder = os.path.join("falcon_imitation_data/", f'update_{checkpoint_index:04d}')
        os.makedirs(update_folder, exist_ok=True)
        print(f"[DEBUG] 创建评估文件夹: {update_folder}")
        
        # 保存到文件，文件名与训练时一致
        save_path = os.path.join(update_folder, f'rollout_data_update_{checkpoint_index}.pt')
        torch.save(imitation_data, save_path)
        print(f"[DEBUG] 评估数据已保存到: {save_path}")
        
        # 保存数据统计信息，与训练时格式一致
        stats_path = os.path.join(update_folder, f'rollout_stats_update_{checkpoint_index}.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Update: {checkpoint_index}\n")
            f.write(f"Num steps: {eval_data_collection['total_steps']}\n")
            f.write(f"Num envs: {imitation_data['num_envs']}\n")
            f.write(f"Observations sensors: {list(imitation_data['observations'].keys())}\n")
            if imitation_data['actions'] is not None:
                f.write(f"Actions shape: {imitation_data['actions'].shape}\n")
            if imitation_data['rewards'] is not None:
                f.write(f"Rewards shape: {imitation_data['rewards'].shape}\n")
                f.write(f"Mean reward: {imitation_data['rewards'].mean().item():.4f}\n")
                f.write(f"Total reward: {imitation_data['rewards'].sum().item():.4f}\n")
            if imitation_data['info_data'] is not None:
                f.write(f"Info data keys: {list(imitation_data['info_data'].keys())}\n")
                for key, value in imitation_data['info_data'].items():
                    f.write(f"  {key} shape: {value.shape}\n")
            if imitation_data['running_episode_stats'] is not None:
                f.write(f"Running episode stats keys: {list(imitation_data['running_episode_stats'].keys())}\n")
                for key, value in imitation_data['running_episode_stats'].items():
                    if hasattr(value, 'shape'):
                        f.write(f"  {key} shape: {value.shape}\n")
                    else:
                        f.write(f"  {key}: {type(value)}\n")
        
        logger.info(f"Evaluation data saved to {save_path}")
        logger.info(f"Evaluation stats saved to {stats_path}")
    
    def _save_imitation_learning_data(self, imitation_learning_data, episode_info, episode_stats, env_idx, checkpoint_index):
        """保存模仿学习数据，按照统一时间戳目录结构保存"""
        
        # 创建主保存目录（使用统一时间戳）
        main_folder = os.path.join("falcon_imitation_data", self._save_timestamp)
        os.makedirs(main_folder, exist_ok=True)
        
        # 创建四个子目录
        jaw_rgb_folder = os.path.join(main_folder, "jaw_rgb_data")
        jaw_depth_folder = os.path.join(main_folder, "jaw_depth_data")
        topdown_folder = os.path.join(main_folder, "topdown_map")
        other_folder = os.path.join(main_folder, "other_data")
        
        os.makedirs(jaw_rgb_folder, exist_ok=True)
        os.makedirs(jaw_depth_folder, exist_ok=True)
        os.makedirs(topdown_folder, exist_ok=True)
        os.makedirs(other_folder, exist_ok=True)
        
        # 提取scene_id的basename（不含路径、不含扩展名）来构造唯一文件名
        scene_name = os.path.splitext(os.path.basename(episode_info.scene_id))[0]
        episode_id_num = int(episode_info.episode_id) if episode_info.episode_id.isdigit() else hash(episode_info.episode_id) % 1000000
        episode_filename = f"{scene_name}_ep{episode_id_num:06d}.pkl"
        
        # 1. 保存RGB图像数据
        if imitation_learning_data['jaw_rgb_data'][env_idx]:
            rgb_data = np.stack(imitation_learning_data['jaw_rgb_data'][env_idx], axis=0)  # (T, H, W, C)
            rgb_dict = {
                "agent_0_articulated_agent_jaw_rgb": rgb_data
            }
            rgb_path = os.path.join(jaw_rgb_folder, episode_filename)
            with open(rgb_path, 'wb') as f:
                pickle.dump(rgb_dict, f)
            print(f"[DEBUG] 保存 RGB 数据: shape = {rgb_data.shape}, episode {scene_name}_ep{episode_id_num:06d} 到: {main_folder}/jaw_rgb_data/")
        else:
            print(f"[DEBUG] 环境 {env_idx} 没有 RGB 数据可保存，episode {scene_name}_ep{episode_id_num:06d}")
        
        # 2. 保存深度图像数据
        if imitation_learning_data['jaw_depth_data'][env_idx]:
            depth_data = np.stack(imitation_learning_data['jaw_depth_data'][env_idx], axis=0)  # (T, H, W, 1)
            depth_dict = {
                "agent_0_articulated_agent_jaw_depth": depth_data
            }
            depth_path = os.path.join(jaw_depth_folder, episode_filename)
            with open(depth_path, 'wb') as f:
                pickle.dump(depth_dict, f)
            print(f"[DEBUG] 保存深度数据: shape = {depth_data.shape}, episode {scene_name}_ep{episode_id_num:06d} 到: {main_folder}/jaw_depth_data/")
        else:
            print(f"[DEBUG] 环境 {env_idx} 没有深度数据可保存，episode {scene_name}_ep{episode_id_num:06d}")
        
        # 3. 保存topdown_map数据（从info_data中提取完整序列）
        topdown_maps = []
        if imitation_learning_data['other_data']['info_data'][env_idx]:
            # 从info_data中提取所有帧的top_down_map
            for info in imitation_learning_data['other_data']['info_data'][env_idx]:
                if isinstance(info, dict) and 'top_down_map' in info:
                    topdown = info['top_down_map']
                    if isinstance(topdown, dict) and 'map' in topdown:
                        # 只保存地图本身，节省空间
                        topdown_maps.append(topdown['map'])
        
        if topdown_maps:
            # 将所有帧的topdown地图堆叠成一个数组
            topdown_sequence = np.stack(topdown_maps, axis=0)  # (T, H, W)
            topdown_dict = {
                "top_down_map": topdown_sequence
            }
            topdown_path = os.path.join(topdown_folder, episode_filename)
            with open(topdown_path, 'wb') as f:
                pickle.dump(topdown_dict, f)
            print(f"[DEBUG] 保存 topdown 序列: shape = {topdown_sequence.shape}, episode {scene_name}_ep{episode_id_num:06d} 到: {main_folder}/topdown_map/")
        else:
            print(f"[DEBUG] 环境 {env_idx} 没有 topdown 数据可保存，episode {scene_name}_ep{episode_id_num:06d}")
        
        # 4. 保存其他数据
        other_data_dict = {}
        
        # 动作数据
        if imitation_learning_data['other_data']['actions'][env_idx]:
            actions = np.stack(imitation_learning_data['other_data']['actions'][env_idx], axis=0)  # (T, action_dim)
            other_data_dict['actions'] = actions
        
        # 奖励数据
        if imitation_learning_data['other_data']['rewards'][env_idx]:
            rewards = np.array(imitation_learning_data['other_data']['rewards'][env_idx], dtype=np.float64)  # (T,)
            other_data_dict['rewards'] = rewards
        
        # Mask数据
        if imitation_learning_data['other_data']['masks'][env_idx]:
            masks = np.array(imitation_learning_data['other_data']['masks'][env_idx], dtype=np.float64)  # (T,)
            other_data_dict['masks'] = masks
        
        # Info数据（移除topdown_map以减少冗余）
        if imitation_learning_data['other_data']['info_data'][env_idx]:
            # 创建不包含topdown_map的info_data副本
            cleaned_info_data = []
            for info in imitation_learning_data['other_data']['info_data'][env_idx]:
                if isinstance(info, dict):
                    # 创建副本并移除top_down_map
                    cleaned_info = {k: v for k, v in info.items() if k != 'top_down_map'}
                    cleaned_info_data.append(cleaned_info)
                else:
                    cleaned_info_data.append(info)
            other_data_dict['info_data'] = cleaned_info_data
        
        # GPS/Compass数据
        if imitation_learning_data['other_data']['pointgoal_with_gps_compass'][env_idx]:
            gps_compass = np.stack(imitation_learning_data['other_data']['pointgoal_with_gps_compass'][env_idx], axis=0)  # (T, 2)
            other_data_dict['agent_0_pointgoal_with_gps_compass'] = gps_compass
        
        # Episode统计信息
        other_data_dict['episode_stats'] = episode_stats
        other_data_dict['scene_id'] = episode_info.scene_id
        other_data_dict['episode_id'] = episode_info.episode_id
        other_data_dict['checkpoint_index'] = checkpoint_index
        
        # 保存其他数据
        other_path = os.path.join(other_folder, episode_filename)
        with open(other_path, 'wb') as f:
            pickle.dump(other_data_dict, f)
        print(f"[DEBUG] 保存 episode {scene_name}_ep{episode_id_num:06d} 到: {main_folder}/other_data/")
        
        # 清空当前环境的数据，为下一个episode做准备
        imitation_learning_data['jaw_rgb_data'][env_idx] = []
        imitation_learning_data['jaw_depth_data'][env_idx] = []
        for key in imitation_learning_data['other_data']:
            imitation_learning_data['other_data'][key][env_idx] = []
        print(f"[DEBUG] 已清空环境 {env_idx} 的模仿学习数据缓存")
    
    def __init__(self, config=None):
        # 初始化时间戳，用于统一的数据保存目录命名
        self._save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 添加数据保存配置
        if config is not None:
            self._save_eval_data = getattr(config.habitat_baselines.eval, 'save_eval_data', True)
            self._save_observations = getattr(config.habitat_baselines.eval, 'save_observations', True)
            self._save_episode_info = getattr(config.habitat_baselines.eval, 'save_episode_info', True)
            self._save_rewards = getattr(config.habitat_baselines.eval, 'save_rewards', True)
            print(f"[DEBUG] FALCONEvaluator 初始化: _save_eval_data = {self._save_eval_data}, timestamp = {self._save_timestamp}")
        else:
            self._save_eval_data = True
            self._save_observations = True
            self._save_episode_info = True
            self._save_rewards = True
            print(f"[DEBUG] FALCONEvaluator 初始化: config=None, _save_eval_data = {self._save_eval_data}, timestamp = {self._save_timestamp}")
