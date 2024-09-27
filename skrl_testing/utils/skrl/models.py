import gym
import gymnasium
import numpy as np
import os
import torch
import torch.nn as nn
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image
from skrl.models.torch import Model  # noqa
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, MultivariateGaussianMixin  # noqa

# from multimodal_gym.utils.image_utils import save_image, save_images_to_file

activations = {
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "identity": nn.Identity(),
}


def build_sequential_network(inputs, hiddens, outputs, hidden_activation, output_activation):
    layers = []

    # First hidden layer: from inputs to the first hidden layer
    layers.append(nn.Linear(inputs, hiddens[0]))
    layers.append(activations[hidden_activation[0]])  # Add activation

    # Hidden layers: loop over hidden layers
    for i in range(len(hiddens) - 1):
        layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
        layers.append(activations[hidden_activation[i + 1]])  # Add activation

    # Output layer: from the last hidden layer to the output
    layers.append(nn.Linear(hiddens[-1], outputs))
    layers.append(activations[output_activation])

    return nn.Sequential(*layers)


class Shape(Enum):
    """
    Enum to select the shape of the model's inputs and outputs
    """

    ONE = 1
    STATES = 0
    OBSERVATIONS = 0
    ACTIONS = -1
    STATES_ACTIONS = -2


def custom_shared_model(
    observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    structure: str = "",
    roles: Sequence[str] = [],
    parameters: Sequence[Mapping[str, Any]] = [],
    single_forward_pass: bool = True,
    frame_stack: int = 1,
    num_gt_observations: int = 4,
) -> Model:
    """Instantiate a shared model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param structure: Shared model structure (default: ``""``).
                      Note: this parameter is ignored for the moment
    :type structure: str, optional
    :param roles: Organized list of model roles (default: ``[]``)
    :type roles: sequence of strings, optional
    :param parameters: Organized list of model instantiator parameters (default: ``[]``)
    :type parameters: sequence of dict, optional
    :param single_forward_pass: Whether to perform a single forward-pass for the shared layers/network (default: ``True``)
    :type single_forward_pass: bool

    :return: Shared model instance
    :rtype: Model
    """

    class GaussianDeterministicModel(GaussianMixin, DeterministicMixin, Model):
        def __init__(
            self,
            observation_space,
            action_space,
            device,
            roles,
            metadata,
            single_forward_pass,
            frame_stack,
            num_gt_observations,
        ):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(
                self,
                clip_actions=metadata[0]["clip_actions"],
                clip_log_std=metadata[0]["clip_log_std"],
                min_log_std=metadata[0]["min_log_std"],
                max_log_std=metadata[0]["max_log_std"],
                role=roles[0],
            )
            DeterministicMixin.__init__(self, clip_actions=metadata[1]["clip_actions"], role=roles[1])

            self._roles = roles
            self._single_forward_pass = single_forward_pass
            self.instantiator_input_type = metadata[0]["input_shape"].value
            self.instantiator_output_scales = [m["output_scale"] for m in metadata]

            print("Observation space:", observation_space)
            num_inputs = observation_space.shape[0]
            num_actions = metadata[1]["output_shape"].value

            self.obs_type = metadata[0]["obs_type"]

            # print(observation_space)
            self.num_gt_observations = num_gt_observations

            num_inputs, self.cnn = process_inputs(
                self.obs_type,
                frame_stack,
                metadata[0]["latent_dim"],
                metadata[0]["img_dim"],
                num_inputs,
                num_gt_observations,
            )

            # shared layers/network
            self.net = nn.Sequential(nn.Linear(num_inputs, 32), nn.ELU(), nn.Linear(32, 32), nn.ELU()).to(device)

            self.mean_net = nn.Sequential(nn.Linear(32, num_actions), nn.Tanh()).to(device)
            self.log_std_parameter = nn.Parameter(torch.zeros(num_actions)).to(device)

            self.value_net = nn.Sequential(nn.Linear(32, 1), nn.Identity()).to(device)

        def act(self, inputs, role):
            if role == self._roles[0]:
                return GaussianMixin.act(self, inputs, role)
            elif role == self._roles[1]:
                return DeterministicMixin.act(self, inputs, role)

        def compute(self, inputs, role):
            if self.instantiator_input_type == 0:
                net_inputs = inputs["states"]
            elif self.instantiator_input_type == -1:
                net_inputs = inputs["taken_actions"]
            elif self.instantiator_input_type == -2:
                net_inputs = torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)

            if self.obs_type == "image":
                # pass input first through cnn
                net_inputs = self.cnn(net_inputs)

            elif self.obs_type == "concat":
                # pass input first through cnn
                prop_obs = net_inputs[:, : self.num_gt_observations]
                img_obs = net_inputs[:, self.num_gt_observations :]
                z = self.cnn(img_obs)
                net_inputs = torch.cat((prop_obs, z), dim=1)


            # single shared layers/network forward-pass
            if self._single_forward_pass:
                if role == self._roles[0]:
                    self._shared_output = self.net(net_inputs)
                    return (
                        self.instantiator_output_scales[0] * self.mean_net(self._shared_output),
                        self.log_std_parameter,
                        {},
                    )
                elif role == self._roles[1]:
                    shared_output = self.net(net_inputs) if self._shared_output is None else self._shared_output
                    self._shared_output = None
                    return self.instantiator_output_scales[1] * self.value_net(shared_output), {}
            # multiple shared layers/network forward-pass
            else:
                raise NotImplementedError

    # TODO: define the model using the specified structure

    return GaussianDeterministicModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        roles=roles,
        metadata=parameters,
        single_forward_pass=single_forward_pass,
        frame_stack=frame_stack,
        num_gt_observations=num_gt_observations,
    )


def process_inputs(obs_type, frame_stack, latent_dim, img_dim, num_inputs, num_gt_observations, num_prop_observations, num_layers, downsample):

    # create cnn if image included
    if obs_type == "image" or obs_type == "image_prop":
        obs_dim = (frame_stack * 3, img_dim, img_dim)
        cnn = ImageEncoder(
            obs_dim, img_dim, frame_stack, feature_dim=latent_dim, num_layers=num_layers, downsample=downsample
        )
    else:
        cnn = None

    if obs_type == "image":
        num_inputs = latent_dim
    elif obs_type == "image_prop":
        num_inputs = latent_dim + num_prop_observations

    return num_inputs, cnn


def custom_gaussian_model(
    observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    initial_log_std: float = 0,
    input_shape: Shape = Shape.STATES,
    hiddens: list = [256, 256],
    hidden_activation: list = ["relu", "relu"],
    output_shape: Shape = Shape.ACTIONS,
    output_activation: Optional[str] = "tanh",
    output_scale: float = 1.0,
    obs_type: str = "prop",
    latent_dim: int = 512,
    img_dim: int = 80,
    num_layers: int = 4,
    downsample: bool = False,
    frame_stack: int = 1,
    num_gt_observations: int = 4,
    num_prop_observations: int = 0,
) -> Model:
    """Instantiate a Gaussian model

    :return: Gaussian model instance
    :rtype: Model
    """

    class GaussianModel(GaussianMixin, Model):
        def __init__(
            self,
            observation_space,
            action_space,
            device,
            clip_actions,
            clip_log_std,
            min_log_std,
            max_log_std,
            frame_stack,
            num_gt_observations,
            reduction="sum",
        ):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            # print("*********Creating policy network*********")
            self.obs_type = obs_type

            num_inputs = observation_space.shape[0]
            num_actions = action_space.shape[0]

            self.net = build_sequential_network(num_inputs, hiddens, num_actions, hidden_activation, output_activation).to(device)

            self.log_std_parameter = nn.Parameter(initial_log_std * torch.ones(num_actions)).to(device)

            print(self.net)

        def compute(self, inputs, role=""):

            net_inputs = inputs["states"]

            output = self.net(net_inputs)

            return output * self.instantiator_output_scale, self.log_std_parameter, {}

    metadata = {
        "input_shape": input_shape,
        "hiddens": hiddens,
        "hidden_activation": hidden_activation,
        "output_shape": output_shape,
        "output_activation": output_activation,
        "output_scale": output_scale,
        "initial_log_std": initial_log_std,
    }

    return GaussianModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        frame_stack=frame_stack,
        num_gt_observations=num_gt_observations,
    )


def custom_deterministic_model(
    observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    input_shape: Shape = Shape.STATES,
    hiddens: list = [256, 256],
    hidden_activation: list = ["relu", "relu"],
    output_shape: Shape = Shape.ACTIONS,
    output_activation: Optional[str] = "tanh",
    output_scale: float = 1.0,
    obs_type: str = "prop",
    latent_dim: int = 512,
    img_dim: int = 80,
    num_layers: int = 4,
    downsample: bool = False,
    frame_stack: int = 1,
    num_gt_observations: int = 4,
    num_prop_observations: int = 0,
) -> Model:
    """Instantiate a deterministic model

    :return: Deterministic model instance
    :rtype: Model
    """

    class DeterministicModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions, frame_stack, num_gt_observations):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            self.obs_type = obs_type

            num_inputs = observation_space.shape[0]
            
            self.net = build_sequential_network(num_inputs, hiddens, 1, hidden_activation, output_activation).to(device)

            print(self.net)

        def compute(self, inputs, role=""):

            net_inputs = inputs["states"]

            output = self.net(net_inputs)

            return output * self.instantiator_output_scale, {}

    metadata = {
        "input_shape": input_shape,
        "hiddens": hiddens,
        "hidden_activation": hidden_activation,
        "output_shape": output_shape,
        "output_activation": output_activation,
        "output_scale": output_scale,
    }

    return DeterministicModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        frame_stack=frame_stack,
        num_gt_observations=num_gt_observations,

    )
