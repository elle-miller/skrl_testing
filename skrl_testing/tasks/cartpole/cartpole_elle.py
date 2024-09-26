# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence

from multimodal_gym.assets.cartpole import CARTPOLE_CFG
from multimodal_gym.utils.image_utils import save_image
from multimodal_gym.utils.isaaclab.direct_rl_env import DirectRLElleEnv

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # physics sim
    physics_dt = 1 / 120
    # number of physics step per control step
    decimation = 2
    # the number of physics simulation steps per rendering steps (default=1)
    render_interval = 2
    episode_length_s = 5.0  # 5 * 120 / 2 = 300 timesteps

    action_scale = 100.0  # [N]
    num_actions = 1
    num_states = 0

    # specific settings
    num_actions = 1
    num_states = 0
    num_gt_observations = 4
    num_prop_observations = 4

    # these will be overwritten in config
    num_observations = 0
    obs_type = "image"

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=render_interval)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.125, 0.125]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

    # camera stuff
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=84,
        height=84,
    )

    # change viewer settings
    viewer = ViewerCfg(eye=(5, 0, 3), lookat=(0, 0, 2), resolution=(1440, 960))

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)

    # occlusions
    occlusion = False
    write_image_to_file = False
    frame_stack = 1
    hw = 80
    random_crop = False

    pole_rgb = (12, 20, 36)
    cart_rgb = (44, 60, 84)


# DirectRLElleEnv
class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg

    def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add camera
        if self.cfg.obs_type == "image" or self.cfg.obs_type == "image_prop":
            self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
            self.scene.sensors["tiled_camera"] = self._tiled_camera

    def get_num_observations(self):
        self.num_img_observations = (
            self.cfg.frame_stack * 3 * self.cfg.tiled_camera.height * self.cfg.tiled_camera.width
        )
        match self.cfg.obs_type:
            case "prop":
                return self.cfg.num_gt_observations
            case "image":
                return self.num_img_observations
            case "image_prop":
                return self.num_img_observations + self.cfg.num_gt_observations
            # case

        raise ValueError("Value doesn't match")

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.get_num_observations()
        self.num_states = self.cfg.num_states

        print("Configured gym env with ", self.num_observations, "observations")

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_observations,))
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:

        if self.cfg.obs_type == "prop":
            obs = self._get_proprioception()
        elif self.cfg.obs_type == "image":
            obs = self._get_images()
        elif self.cfg.obs_type == "image_prop":
            prop = self._get_proprioception()
            imgs = self._get_images()
            obs = torch.cat((prop, imgs), dim=-1)
        else:
            print("Unknown observations type!")

        return obs

    def _get_rewards(self) -> torch.Tensor:
        total_reward, rew_alive, rew_termination, rew_pole_pos, rew_cart_vel, rew_pole_vel = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        self.extras["log"] = {
            "rew_alive": (rew_alive).mean(),
            "rew_termination": (rew_termination).mean(),
            "rew_pole_pos": (rew_pole_pos).mean(),
            "rew_cart_vel": (rew_cart_vel).mean(),
            "rew_pole_vel": (rew_pole_vel).mean(),
        }
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _get_proprioception(self):
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        return obs

    def _get_images(self):
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"

        # returned data will be (num_cameras, height, width, num_channels) ((1024, 80, 80, 3)) torch.float32
        img_batch = self._tiled_camera.data.output[data_type].clone()
        batch_size = img_batch.size()[0]

        # make bg white
        img_batch[img_batch == 0] = 1

        # make half white
        # Calculate the midpoint of the height
        midpoint = 50

        # Make the top half of each image white
        # img_batch[:, :midpoint, :, :] = 255

        if self.cfg.occlusion:
            # get updated disc masks
            disc_img, disc_mask = self.discs.get_image()
            disc_img = (disc_img.astype(np.float32)) / 255

            # mask the image batch
            disc_img_tensor = (
                torch.tensor(disc_img, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(self.sim.device)
            )
            disc_mask_tensor = (
                torch.tensor(disc_mask, dtype=torch.bool).unsqueeze(2).repeat(batch_size, 1, 1, 3).to(self.sim.device)
            )
            img_batch[disc_mask_tensor] = disc_img_tensor[disc_mask_tensor]

        if self.cfg.write_image_to_file:
            save_image(img_batch, img_name="cartpole.png", nchw=False)

        flattened_images = img_batch.view(batch_size, -1)

        return flattened_images


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward, rew_alive, rew_termination, rew_pole_pos, rew_cart_vel, rew_pole_vel
