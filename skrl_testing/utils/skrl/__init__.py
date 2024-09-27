import gym
import gymnasium
from typing import Any, Union

from skrl import logger
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper

from skrl_testing.utils.skrl.base_wrapper import Wrapper
from skrl_testing.utils.skrl.isaaclab_wrapper import IsaacLabWrapper

__all__ = ["wrap_env", "Wrapper", "MultiAgentEnvWrapper"]


def wrap_env(
    env: Any, wrapper: str = "auto", verbose: bool = True, obs_type=None
) -> Union[Wrapper, MultiAgentEnvWrapper]:
    """Wrap an environment to use a common interface

    Example::

        >>> from skrl.envs.wrappers.torch import wrap_env
        >>>
        >>> # assuming that there is an environment called "env"
        >>> env = wrap_env(env)

    :param env: The environment to be wrapped
    :type env: gym.Env, gymnasium.Env, dm_env.Environment or VecTask
    :param wrapper: The type of wrapper to use (default: ``"auto"``).
                    If ``"auto"``, the wrapper will be automatically selected based on the environment class.
                    The supported wrappers are described in the following table:

                    +--------------------+-------------------------+
                    |Environment         |Wrapper tag              |
                    +====================+=========================+
                    |OpenAI Gym          |``"gym"``                |
                    +--------------------+-------------------------+
                    |Gymnasium           |``"gymnasium"``          |
                    +--------------------+-------------------------+
                    |Petting Zoo         |``"pettingzoo"``         |
                    +--------------------+-------------------------+
                    |DeepMind            |``"dm"``                 |
                    +--------------------+-------------------------+
                    |Robosuite           |``"robosuite"``          |
                    +--------------------+-------------------------+
                    |Bi-DexHands         |``"bidexhands"``         |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 2 |``"isaacgym-preview2"``  |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 3 |``"isaacgym-preview3"``  |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 4 |``"isaacgym-preview4"``  |
                    +--------------------+-------------------------+
                    |Omniverse Isaac Gym |``"omniverse-isaacgym"`` |
                    +--------------------+-------------------------+
                    |Isaac Lab           |``"isaaclab"``           |
                    +--------------------+-------------------------+
    :type wrapper: str, optional
    :param verbose: Whether to print the wrapper type (default: ``True``)
    :type verbose: bool, optional

    :raises ValueError: Unknown wrapper type

    :return: Wrapped environment
    :rtype: Wrapper or MultiAgentEnvWrapper
    """

    def _get_wrapper_name(env, verbose):
        def _in(value, container):
            for item in container:
                if value in item:
                    return True
            return False

        base_classes = [str(base).replace("<class '", "").replace("'>", "") for base in env.__class__.__bases__]
        try:
            base_classes += [
                str(base).replace("<class '", "").replace("'>", "") for base in env.unwrapped.__class__.__bases__
            ]
        except:
            pass
        base_classes = sorted(list(set(base_classes)))
        if verbose:
            logger.info(f"Environment wrapper: 'auto' (class: {', '.join(base_classes)})")

        if _in("omni.isaac.lab.envs.manager_based_env.ManagerBasedEnv", base_classes) or _in(
            "omni.isaac.lab.envs.direct_rl_env.DirectRLEnv", base_classes
        ):
            return "isaaclab"

        return base_classes

    if wrapper == "auto":
        wrapper = _get_wrapper_name(env, verbose)

    if wrapper == "isaaclab" or wrapper == "isaac-orbit":
        if verbose:
            logger.info("Environment wrapper: Isaac Lab")
        return IsaacLabWrapper(env)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper}")
