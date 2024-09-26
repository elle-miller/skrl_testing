from typing import List, Optional, Union

import copy
import sys
import tqdm

import torch
from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
# from skrl.trainers.torch import Trainer


# [start-config-dict-torch]
SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",  # key used to get and log environment info
}
# [end-config-dict-torch]

class Trainer:
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)
        self.environment_info = self.cfg.get("environment_info", "episode")

        self.initial_timestep = 0

        # Using this trainer in an alternating fashion (e.g., .train() -> .eval() -> .train()) will restart the
        # env on each call to .train(). This is an issue if we are using the SKRL memory class, as it is not
        # aware of manual restarts. This means we will be storing a trajectory that will randomly have resets
        # *without* a DONE flag. This can cause learning instabilities. Therefore, ONLY CALL .reset() ONCE!
        # self.started_already = False

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:
            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")

        # update trainer configuration to avoid duplicated info/data in distributed runs
        if config.torch.is_distributed:
            if config.torch.rank:
                self.disable_progressbar = True

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})")
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})")
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})")
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError
    

class SequentialTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Sequential trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See SEQUENTIAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        # init agents
        self.agents.init(trainer_cfg=self.cfg)
        self.training_timestep = 0

    def train(self, train_timesteps) -> None:
        """Train the agents sequentially for train_timesteps

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        self.agents.set_running_mode("train")

        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        # HARD reset of all environments to begin evaluation
        states, infos = self.env.reset() # hard=True)

        # Resetting here helps with .train()->.eval()->.train() The first .train() rollout could be interrupted by
        # the call to .eval(). This interruption is likely not recorded in the memory, so the training stage
        # may compute information across trajectories, which is not ideal.
        # We also need to reset the agent's "_rollout" attribute, as this determines when the agent is actually
        # updated. Resetting it here ensures that each agent update happens with the hyperparam-specified
        # frequency.
        self.agents.memory.reset()
        self.agents._rollout = 0

        train_start = self.training_timestep
        train_pause = self.training_timestep + train_timesteps

        # print(train_start, train_pause)

        for timestep in tqdm.tqdm(range(train_start, train_pause), disable=self.disable_progressbar, file=sys.stdout):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

            self.training_timestep += 1


    def eval(self, record=False) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        
        self.agents.set_running_mode("eval")

        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

       # Hard reset all environments
        states, infos = self.env.reset() #hard=True)

        # TODO: integrate the actual total returns from the env
        infos = {'reach_reward': None, 'lift_reward': None, 'object_goal_tracking': None, 'action_rate': None,
                 'joint_vel_penalty': None, 'reach_success': None, 'returns': None, 'returns_masked': None}

        returns = {k: torch.zeros(size=(states.shape[0], 1), device=states.device) for k in infos.keys()}
        mask = torch.Tensor([[1] for _ in range(states.shape[0])]).to(states.device)
        term_mask = torch.Tensor([[1] for _ in range(states.shape[0])]).to(states.device)
        trunc_mask = torch.Tensor([[1] for _ in range(states.shape[0])]).to(states.device)
        steps_to_term = torch.Tensor([[0] for _ in range(states.shape[0])]).to(states.device)
        returns['steps_to_term'] = steps_to_term
        returns['steps_to_trunc'] = steps_to_term.clone()
        ep_length = self.env.env.max_episode_length - 1

        termination_counter = torch.zeros(size=(states.shape[0], 1)).to(states.device)
        truncated_counter = torch.zeros(size=(states.shape[0], 1)).to(states.device)

        images = []

        for timestep in tqdm.tqdm(range(self.initial_timestep, ep_length), disable=self.disable_progressbar, file=sys.stdout):

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=ep_length, eval=True)[0]

                # step the environments
                next_states, rewards, truncated, terminated, infos = self.env.step(actions)
                
                mask_update = 1 - torch.logical_or(terminated, truncated).float()

                termination_counter += terminated
                truncated_counter += truncated

                self.env.unwrapped.common_step_counter -= 1

                if 'log' in infos:
                    for k, v in infos['log'].items():
                        if k in returns:
                            returns[k] += v * mask
                returns['returns'] += rewards #* mask
                returns['returns_masked'] += rewards * mask
                returns['steps_to_term'] += torch.ones_like(mask, device=mask.device) * term_mask
                returns['steps_to_trunc'] += torch.ones_like(mask, device=mask.device) * trunc_mask
                mask *= mask_update
                term_mask *= (1 - terminated.float())
                trunc_mask *= (1 - truncated.float())
                
                if record:
                    images.append(next_states[0])

                # render scene
                if not self.headless:
                    self.env.render()
                    
            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

        returns['reached_timesteps'] = returns['reach_success']
        returns['reach_success'] = (returns['reach_success'] > 1.0).float()
        
        return returns, images
     