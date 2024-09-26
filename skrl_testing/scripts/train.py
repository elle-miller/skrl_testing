# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys

from omni.isaac.lab.app import AppLauncher
import numpy as np
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=9999999999, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument('--run_notes', default=None, type=str, help='notes for the run')
parser.add_argument("--record", action="store_true", default=False, help="Record videos during eval.")
parser.add_argument('--learning_epochs', type=int, default=8)
parser.add_argument('--mini_batches', type=int, default=8)
parser.add_argument('--timesteps', type=int, default=5_000, help='iters per .train() call')
parser.add_argument("--random_crop", action="store_true", default=False, help="Whether to use random crop augmentation ala DrQ.")

parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument("--frame_stack", type=int, action="store", default=1, help="Choose from static, moving")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import traceback
from datetime import datetime
import torch
import numpy as np
import carb
import skrl

import wandb

os.environ['WANDB_DIR'] = './wandb'
os.environ['WANDB_CACHE_DIR'] = './wandb'
os.environ['WANDB_CONFIG_DIR'] = './wandb'
os.environ['WANDB_DATA_DIR'] = './wandb'



from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
# from skrl.utils.model_instantiators.torch import gaussian_model as custom_gaussian_model
# from skrl.utils.model_instantiators.torch import deterministic_model as custom_deterministic_model
# from skrl.utils.model_instantiators.torch import shared_model as custom_shared_model
from skrl_testing.utils.skrl.models import custom_gaussian_model, custom_deterministic_model, custom_shared_model
# from skrl.agents.torch.ppo.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
# from skrl.trainers.torch.sequential import SequentialTrainer

import skrl_testing.tasks  # noqa: F401
from skrl_testing.utils.skrl.skrl_wrapper import process_skrl_cfg
from skrl_testing.utils.skrl.sequential import SequentialTrainer
from skrl_testing.utils.config import LOG_ROOT_DIR
from skrl_testing.utils.skrl.ppo import PPO, PPO_DEFAULT_CONFIG

# Import extensions to set up environment tasks
# import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    """Train with skrl agent."""
    skrl_policy_config_dict = process_skrl_cfg(agent_cfg["models"]["policy"], ml_framework=args_cli.ml_framework)
    skrl_value_config_dict = process_skrl_cfg(agent_cfg["models"]["value"], ml_framework=args_cli.ml_framework)
    skrl_shared_config_dict = process_skrl_cfg(agent_cfg["models"]["shared"], ml_framework=args_cli.ml_framework)


    # override configurations with either config file or args
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    set_seed(args_cli.seed if args_cli.seed is not None else agent_cfg["seed"])

    # specify directory for logging experiments
    log_root_path = os.path.join(LOG_ROOT_DIR, "logs", "skrl", agent_cfg["agent"]["experiment"]["directory"], agent_cfg["agent"]["experiment"]["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specific run  {time-stamp}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        # Should we only record during eval? Probably...
        env = gym.wrappers.RecordVideo(env, **video_kwargs)  

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    models = {}
    # non-shared models
    if agent_cfg["models"]["separate"]:
        models["policy"] = custom_gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **skrl_policy_config_dict,
            **skrl_shared_config_dict
        )
        models["value"] = custom_deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **skrl_value_config_dict,
            **skrl_shared_config_dict
        )
    # shared models
    else:
        models["policy"] = custom_shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                skrl_policy_config_dict,
                skrl_value_config_dict,
                skrl_shared_config_dict
            ],
        )
        models["value"] = models["policy"]

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    memory_size = agent_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device) #, env_cfg=env_cfg)

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
    default_agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    default_agent_cfg.update(process_skrl_cfg(agent_cfg["agent"], ml_framework=args_cli.ml_framework))
    default_agent_cfg["state_preprocessor"] = None
    default_agent_cfg["value_preprocessor"] = RunningStandardScaler
    default_agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    default_agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * agent_cfg["rewards_shaper_scale"]

    agent = PPO(
        models=models,
        memory=memory,
        cfg=default_agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    max_timesteps = agent_cfg["trainer"]["max_timesteps"]
    num_eval = agent_cfg["trainer"]["num_eval"]
    train_timesteps = int(max_timesteps // num_eval)
    agent_cfg["trainer"]["timesteps"] = train_timesteps

    # configure and instantiate a custom RL trainer for logging episode events
    trainer_cfg = agent_cfg["trainer"]
    trainer_cfg["close_environment_at_exit"] = False
    trainer_cfg["disable_progressbar"] = False
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
   
   # setup wandb
    if agent_cfg["agent"]["experiment"]["wandb"] == True:
        print("using wandb........")
        wandb.init(
            project=agent_cfg["agent"]["experiment"]["wandb_kwargs"]["project"],
            entity='my-phd',
            group=agent_cfg["agent"]["experiment"]["wandb_kwargs"]["group"],
            name=agent_cfg["agent"]["experiment"]["wandb_kwargs"]["name"],
            config=default_agent_cfg
        )

    record = False
    for step in range(num_eval):
        
        # if record:
        #     returns, images = trainer.eval(True)
        # else:

        global_step = step * train_timesteps

        returns, _ = trainer.eval()

        # assuming agent is an instance of an Agent subclass
        agent.writer.add_scalar("Eval / Returns", returns['returns'].mean().cpu(), global_step=global_step)

        trainer.train(train_timesteps)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()