import torch
from typing import Any, Tuple

# from skrl.envs.wrappers.torch.base import Wrapper
from skrl_testing.utils.skrl.base_wrapper import Wrapper

import gymnasium.spaces as spaces



class IsaacLabWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Lab environment wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Lab environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None
        # self._observation_space = self._observation_space
        self._observation_space = self._observation_space
        # self.env_cfg = env_cfg

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, reward, terminated, truncated, info = self._env.step(actions)

        # flatten obs dict
        # flattened = spaces.flatten(self.env.observation_space, self._obs_dict)
        # print(flattened.size())

        # convert LazyFrames to tensor
        # self._obs_dict = self.convert_to_tensor(self._obs_dict)

        return self._obs_dict, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def reset(self, hard: bool = False) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if hard:
            self._obs_dict, info = self._env.reset()
        else:
            if self._reset_once:
                self._obs_dict, info = self._env.reset()
                self._reset_once = False
            else:
                info = None

        # flatten obs dict
        # flattened = spaces.flatten(self.env.observation_space, self._obs_dict)
        # print(flattened.size())

        # self._obs_dict = self.convert_to_tensor(self._obs_dict)

        return self._obs_dict, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        self._env.close()
