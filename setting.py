# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import argparse
import os
import platform

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('Agg')


def _set_gpu_preallocation(mode: float):
    """Set JAX GPU memory preallocation fraction.

    This function configures how much of the total GPU memory JAX will preallocate
    by setting the XLA_PYTHON_CLIENT_MEM_FRACTION environment variable. By default,
    JAX preallocates 75% of GPU memory, but this can be adjusted to avoid out-of-memory
    errors.

    Args:
        mode: A float between 0.0 and 1.0 (exclusive) representing the fraction of
              GPU memory to preallocate. For example, 0.5 would allocate 50% of
              available GPU memory.

    Raises:
        AssertionError: If mode is not a float or not in the range [0.0, 1.0).

    Example:
        >>> _set_gpu_preallocation(0.5)  # Allocate 50% of GPU memory
    """
    assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


def _set_gpu_device(device_ids):
    """Set visible GPU devices for CUDA and turn off JAX traceback filtering.

    This function configures which GPU devices are visible to the application by setting
    the CUDA_VISIBLE_DEVICES environment variable. It handles different input types
    for device specification.

    Args:
        device_ids: GPU device identifiers in one of these formats:
            - int: Single GPU ID (e.g., 0 for first GPU)
            - tuple/list: Multiple GPU IDs (e.g., [0,1,2] for first three GPUs)
            - str: Comma-separated GPU IDs (e.g., "0,1,2")

    Raises:
        ValueError: If device_ids is not int, tuple, list, or str.

    Note:
        Also disables JAX traceback filtering to show complete stack traces.
    """
    if isinstance(device_ids, int):
        device_ids = str(device_ids)
    elif isinstance(device_ids, (tuple, list)):
        device_ids = ','.join([str(d) for d in device_ids])
    elif isinstance(device_ids, str):
        device_ids = device_ids
    else:
        raise ValueError
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'


class Setting:
    def __init__(self, setting: dict):
        super().__init__()

        self.setting = setting
        self.flywire_version: str = setting['flywire_version']
        self.vjp_method: str = setting['vjp_method']
        self.etrace_decay: float = setting['etrace_decay']
        self.fitting_target: str = setting.get('fitting_target', 'csr')
        self.neural_activity_id: str = setting['neural_activity_id']
        self.neural_activity_max_fr: float = setting['neural_activity_max_fr']
        self.epoch: int = setting['epoch']
        self.duration_per_data = setting['duration_per_data']
        self.duration_per_fr = setting['duration_per_fr']
        self.n_fr_per_warmup = setting['n_fr_per_warmup']
        self.n_fr_per_train = setting['n_fr_per_train']
        self.n_fr_per_gap = setting['n_fr_per_gap']
        self.input_noise_sigma = setting['input_noise_sigma']
        self.lr: float = setting['lr']
        self.weight_mask: int = setting.get('weight_mask', 1)
        self.split: float = setting['split']
        self.dt: float = setting['dt']
        self.loss: str = setting['loss']
        self.input_method: str = setting['input_method']
        self.batch_size: int = setting.get('batch_size', 128)
        self.n_lora_rank: int = setting.get('n_lora_rank', None)
        self.tau_ref: int = setting.get('tau_ref', None)

    def to_dict(self) -> dict:
        """
        Convert the configuration parameters to a dictionary.

        Returns
        -------
        dict
            A dictionary containing all configuration parameters.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'setting'}

    def __repr__(self):
        return repr({k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'setting'})

    def to_filepath(self):
        """
        Convert the configuration parameters to a standardized filepath string.

        Creates a filepath string by concatenating all configuration parameters with '#' delimiters,
        converting quantities to their decimal representations with appropriate units.

        Returns
        -------
        str
            A filepath string containing all parameters in a standardized format:
            'results/flywire_version#neural_activity_id#max_fr#etrace_decay#loss_fn#...'
        """
        filepath = (
            f'results/'
            f'{self.flywire_version}#'
            f'{self.vjp_method}#'
            f'{self.neural_activity_id}#'
            f'{self.neural_activity_max_fr}#'
            f'{self.etrace_decay}#'
            f'{self.loss}#'
            f'{self.input_method}#'
            f'{self.epoch}#'
            f'{self.lr}#'
            f'{self.split}#'
            f'{self.fitting_target}#'
            f'{self.duration_per_data}#'
            f'{self.duration_per_fr}#'
            f'{self.n_fr_per_warmup}#'
            f'{self.n_fr_per_train}#'
            f'{self.n_fr_per_gap}#'
            f'{self.input_noise_sigma}#'
            f'{self.weight_mask}#'
            f'{self.dt}#'
            f'{self.tau_ref}#'
        )
        if self.fitting_target == 'lora':
            filepath += f'{self.n_lora_rank}#'
        # filepath += f'#{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
        return filepath

    @classmethod
    def from_filepath(cls, filepath):
        """
        Parse a filepath string to extract model configuration parameters.

        This method extracts all configuration parameters from a delimited filepath string,
        converts them to appropriate types, and creates a new FilePath instance.

        Parameters
        ----------
        filepath : str
            A filepath string in the format generated by to_filepath().
            Expected format: 'path/to/results/flywire_version#neural_activity_id#max_fr#etrace_decay#...'

        Returns
        -------
        Setting
            A new FilePath instance with parameters extracted from the filepath.
        """
        splits = filepath.split('/')[1].split('#')
        (
            flywire_version,
            vjp_method,
            neural_activity_id,
            neural_activity_max_fr,
            etrace_decay,
            loss,
            input_method,
            epoch,
            lr,
            split,
            fitting_target,
            duration_per_data,
            duration_per_fr,
            n_fr_per_warmup,
            n_fr_per_train,
            n_fr_per_gap,
            input_noise_sigma,
            weight_mask,
            dt,
            tau_ref,
        ) = splits[:-1]
        setting = locals()

        if setting['fitting_target'] == 'lora':
            setting['n_lora_rank'] = int(splits[-1])
        return cls(setting)


def get_parser(gpu_pre_allocate=0.99) -> Setting:
    """Create and configure an argument parser for neural network training settings.

    This function initializes an ArgumentParser with default parameters for training
    neural networks, particularly focusing on GPU settings, dataset configurations,
    and network architecture parameters. It handles two initialization modes:
    - Loading parameters from a specified file
    - Using built-in defaults

    Args:
        gpu_pre_allocate (float, optional): Fraction of GPU memory to pre-allocate.
            Must be between 0.0 and 1.0. Defaults to 0.99.

    Returns:
        argparse.Namespace: Parsed command-line arguments with configured defaults.

    Note:
        This function automatically configures GPU devices and memory allocation
        based on the provided settings.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
    parser.add_argument("--filepath", type=str, default=None)
    args, _ = parser.parse_known_args()

    # device management
    _set_gpu_device(args.devices)
    _set_gpu_preallocation(gpu_pre_allocate)

    if args.filepath is not None:
        with open(os.path.join(args.filepath, 'first-round-losses.txt'), 'r') as f:
            line = f.readline().replace('Namespace', 'dict')
            settings = eval(line)

    else:
        settings = dict(
            vjp_method='multi-step',
            etrace_decay=0.99,
            fitting_target='csr',
            flywire_version='630',
            neural_activity_id='2017-10-26_1',
            neural_activity_max_fr=100.,
            split=0.7,
            lr=1e-4,
            epoch=200,
            loss='mae',
            input_method='current',
            batch_size=64,
            n_lora_rank=20,
            duration_per_data=20.,
            duration_per_fr=20.,
            n_fr_per_warmup=100,
            n_fr_per_train=100,
            n_fr_per_gap=10,
            input_noise_sigma=0.,
            weight_mask=1,
            dt=0.2,
            tau_ref=None,
        )
    default = Setting(settings)

    # training method
    parser.add_argument("--vjp_method", type=str, default=default.vjp_method, choices=['multi-step', 'single-step'],
                        help="The method for computing the Jacobian-vector product (JVP).")
    parser.add_argument("--etrace_decay", type=float, default=default.etrace_decay, help="The etrace time constant.")
    parser.add_argument("--fitting_target", type=str, default=default.fitting_target, choices=['lora', 'csr'])
    args, _ = parser.parse_known_args()
    if args.fitting_target == 'lora':
        parser.add_argument('--n_lora_rank', type=int, default=default.n_lora_rank)

    # training parameters
    parser.add_argument('--dt', type=float, default=default.dt, help='Control the time step of the simulation.')
    parser.add_argument('--duration_per_data', type=float, default=default.duration_per_data)
    parser.add_argument('--duration_per_fr', type=float, default=default.duration_per_fr)
    parser.add_argument('--input_noise_sigma', type=float, default=default.input_noise_sigma)
    parser.add_argument('--n_fr_per_warmup', type=int, default=default.n_fr_per_warmup)
    parser.add_argument('--n_fr_per_train', type=int, default=default.n_fr_per_train)
    parser.add_argument('--n_fr_per_gap', type=int, default=default.n_fr_per_gap)
    parser.add_argument('--epoch', type=int, default=default.epoch, help='The number of training epochs.')
    parser.add_argument('--flywire_version', type=str, default=default.flywire_version, help='The version of flywire.')
    parser.add_argument('--input_method', type=str, default=default.input_method)
    parser.add_argument('--neural_activity_id', type=str, default=default.neural_activity_id, help='The flywire id.')
    parser.add_argument('--neural_activity_max_fr', type=float, default=default.neural_activity_max_fr,
                        help='The maximum firing rate of neural activity. [Hz]')
    parser.add_argument('--lr', type=float, default=default.lr, help='The learning rate.')
    parser.add_argument('--split', type=float, default=default.split, help='The split ratio of training and val set.')
    parser.add_argument('--loss', type=str, default=default.loss, choices=['mse', 'mae'],
                        help='The loss function for training. [mse, mae, huber, cosine_distance, log_cosh]')
    parser.add_argument('--batch_size', type=int, default=default.batch_size, help='The batch size for training.')
    parser.add_argument('--weight_mask', type=int, default=default.weight_mask, help='weight masking.')
    parser.add_argument('--tau_ref', type=lambda x: None if x == 'None' else float(x), default=default.tau_ref)

    return Setting(parser.parse_args().__dict__)
