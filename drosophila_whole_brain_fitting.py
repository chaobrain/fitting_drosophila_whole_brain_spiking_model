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

"""

1、连续数值的 firing rate tokenization

使用 uniform binning，将连续数值的 firing rate tokenize为 若干个 bin，每个 bin 代表 0.1 mV 的 firing rate。
每个 bin 对应一个 token，token 之间的距离为 0.1 mV。每个bin使用随机的 embedding 表示。

2、训练范式

使用上一个时刻的 firing rate 作为输入，当前时刻的 firing rate 作为输出。

"""

from __future__ import annotations

import datetime
import os
import platform
import time
from pathlib import Path
from typing import Callable

from tqdm import tqdm

if platform.system() == 'Linux':
    import matplotlib

    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainevent
import brainstate
import braintools
import brainscale
import brainunit as u
import jax
import jax.numpy as jnp

from utils import g_init, v_init, count_pre_post_connections, barplot, output

base_fn = './data'


class Population(brainstate.nn.Neuron):
    """
    A population of neurons with leaky integrate-and-fire dynamics.

    The dynamics of the neurons are given by the following equations::

       dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
       dg/dt = -g / tau               : volt (unless refractory)


    Args:
      v_rest: The resting potential of the neurons.
      v_reset: The reset potential of the neurons after a spike.
      v_th: The threshold potential of the neurons for spiking.
      tau_m: The membrane time constant of the neurons.
      tau_syn: The synaptic time constant of the neurons.
      tau_ref: The refractory period of the neurons.
      spk_fun: The spike function of the neurons.
      name: The name of the population.

    """

    def __init__(
        self,
        flywire_version: str | int = '783',
        v_rest: u.Quantity = 0 * u.mV,  # resting potential
        v_reset: u.Quantity = 0 * u.mV,  # reset potential after spike
        v_th: u.Quantity = 1 * u.mV,  # potential threshold for spiking
        tau_m: u.Quantity = 20 * u.ms,  # membrane time constant
        # Jürgensen et al https://doi.org/10.1088/2634-4386/ac3ba6
        tau_syn: u.Quantity = 5 * u.ms,  # synaptic time constant
        # Lazar et al https://doi.org/10.7554/eLife.62362
        tau_ref: u.Quantity | None = 2.2 * u.ms,  # refractory period
        spk_fun: Callable = brainstate.surrogate.ReluGrad(width=1.5),  # spike function
        V_init: Callable = brainstate.init.Constant(0 * u.mV),  # initial voltage
        g_init: Callable = brainstate.init.Constant(0. * u.mV),  # initial voltage
        name: str = None,
    ):
        # connectome data
        if flywire_version in ['783', 783]:
            path_neu = os.path.join(base_fn, 'Completeness_783.csv')
            path_syn = os.path.join(base_fn, 'Connectivity_783.parquet')
        elif flywire_version in ['630', 630]:
            path_neu = os.path.join(base_fn, 'Completeness_630_final.csv')
            path_syn = os.path.join(base_fn, 'Connectivity_630_final.parquet')
        else:
            raise ValueError('flywire_version must be either "783" or "630"')
        self.flywire_version = flywire_version

        self.path_neu = Path(path_neu)
        self.path_syn = Path(path_syn)

        print('Loading neuron information ...')

        # neuron ids
        flywire_ids = pd.read_csv(self.path_neu, index_col=0)
        self.n_neuron = len(flywire_ids)

        super().__init__(self.n_neuron, name=name)

        self.flyid2i = {f: i for i, f in enumerate(flywire_ids.index)}
        self.i2flyid = {i: f for i, f in enumerate(flywire_ids.index)}

        # parameters
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        self.tau_m = tau_m
        self.tau_syn = tau_syn
        self.tau_ref = tau_ref if tau_ref is None else u.math.full(self.varshape, tau_ref)
        self.spk_fun = spk_fun

        # initializer
        self.V_init = V_init
        self.g_init = g_init

    def init_state(self):
        self.v = brainscale.ETraceState(brainstate.init.param(self.V_init, self.varshape))
        self.g = brainscale.ETraceState(brainstate.init.param(self.g_init, self.varshape))
        self.spike_count = brainscale.ETraceState(jnp.zeros(self.varshape, dtype=brainstate.environ.dftype()))
        if self.tau_ref is not None:
            self.t_ref = brainstate.ShortTermState(
                brainstate.init.param(brainstate.init.Constant(-1e7 * u.ms), self.varshape)
            )

    def reset_state(self):
        self.reset_spk_count()
        self.v.value = brainstate.init.param(self.V_init, self.varshape)
        if self.tau_ref is not None:
            self.t_ref.value = brainstate.init.param(brainstate.init.Constant(-1e7 * u.ms), self.varshape)

    def reset_spk_count(self, batch_size=None):
        self.spike_count.value = brainstate.init.param(jnp.zeros, self.varshape, batch_size)

    def get_refractory(self):
        if self.tau_ref is None:
            return jnp.zeros(self.varshape, dtype=bool)
        else:
            t = brainstate.environ.get('t')
            ref = (t - self.t_ref.value) <= self.tau_ref
            return ref

    def get_spike(self, v=None):
        v = self.v.value if v is None else v
        return self.spk_fun((v - self.v_th) / (1. * u.mV))

    def update(self, x: u.Quantity[u.mV]):
        t = brainstate.environ.get('t')

        # numerical integration
        dg = lambda g, t: -g / self.tau_syn
        dv = lambda v, t, g: (self.v_rest - v + g) / self.tau_m
        g = brainstate.nn.exp_euler_step(dg, self.g.value, t)
        g += x  # external input current
        v = brainstate.nn.exp_euler_step(dv, self.v.value, t, g)
        v = self.sum_delta_inputs(v)

        # # numerical integration
        # dg = lambda g, t: -g / self.tau_syn
        # dv = lambda v, t, g: (self.v_rest - v + g) / self.tau_m
        # g = brainstate.nn.exp_euler_step(dg, self.g.value, t)
        # v = brainstate.nn.exp_euler_step(dv, self.v.value, t, self.g.value)
        # v = self.sum_delta_inputs(v)
        # g += x  # external input current

        # refractory period
        if self.tau_ref is not None:
            ref = (t - self.t_ref.value) <= self.tau_ref
            v = u.math.where(ref, self.v.value, v)
            g = u.math.where(ref, self.g.value, g)

        # spikes
        spk = self.get_spike(v)
        self.spike_count.value += spk

        # update states
        spk_current = jax.lax.stop_gradient(spk)
        self.v.value = spk_current * (self.v_reset - v) + v
        self.g.value = g - spk_current * g
        if self.tau_ref is not None:
            self.t_ref.value = u.math.where(spk, t, self.t_ref.value)
        return spk


def load_syn(flywire_version: str | int) -> brainevent.CSR:
    if flywire_version in ['783', 783]:
        path_neu = os.path.join(base_fn, 'Completeness_783.csv')
        path_syn = os.path.join(base_fn, 'Connectivity_783.parquet')
    elif flywire_version in ['630', 630]:
        path_neu = os.path.join(base_fn, 'Completeness_630_final.csv')
        path_syn = os.path.join(base_fn, 'Connectivity_630_final.parquet')
    else:
        raise ValueError('flywire_version must be either "783" or "630"')

    # neuron information
    flywire_ids = pd.read_csv(path_neu, index_col=0)
    n_neuron = len(flywire_ids)

    # synapses: CSR connectivity matrix
    flywire_conns = pd.read_parquet(path_syn)
    i_pre = flywire_conns.loc[:, 'Presynaptic_Index'].values
    i_post = flywire_conns.loc[:, 'Postsynaptic_Index'].values
    weight = flywire_conns.loc[:, 'Excitatory x Connectivity'].values
    sort_indices = np.argsort(i_pre)
    i_pre = i_pre[sort_indices]
    i_post = i_post[sort_indices]
    weight = weight[sort_indices]

    values, counts = np.unique(i_pre, return_counts=True)
    indptr = np.zeros(n_neuron + 1, dtype=int)
    indptr[values + 1] = counts
    indptr = np.cumsum(indptr)
    indices = i_post

    csr = brainevent.CSR(
        (np.asarray(weight, dtype=np.float32), indices, indptr),
        shape=(n_neuron, n_neuron)
    )
    return csr


class Interaction(brainstate.nn.Module):
    """
    A neural network module that manages the interaction between a population of neurons,
    external inputs, and synaptic connections.

    Args:
        pop (Population): The population of neurons in the network.
        pop_input (PopulationInput): The input module for the neuron population.
        csr (brainevent.CSR, optional): The sparse connectivity matrix. Defaults to None.
        conn_mode (str, optional): The mode of connectivity, either 'sparse' or 'sparse+low+rank'. Defaults to 'sparse'.
        conn_param_type (type, optional): The type of connection parameters. Defaults to brainscale.ETraceParam.
    """

    def __init__(
        self,
        pop: Population,
        scale_factor: u.Quantity,
        conn_mode: str = 'sparse+low+rank',
        conn_param_type: type = brainscale.ETraceParam,
        n_rank: int = 20,
    ):
        super().__init__()

        # neuronal and synaptic dynamics
        self.pop = pop

        # delay for changes in post-synaptic neuron
        # Paul et al 2015 doi: https://doi.org/10.3389/fncel.2015.00029
        self.delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct(self.pop.varshape, brainstate.environ.dftype()),
            entries={'D': 1.8 * u.ms}
        )

        print('Loading synapse information ...')
        csr = load_syn(self.pop.flywire_version)

        # connectivity matrix
        self.conn_mode = conn_mode
        self.scale_factor = scale_factor

        if conn_mode == 'sparse+low+rank':
            # do not train sparse connection
            self.conn = brainstate.nn.SparseLinear(csr, b_init=None, param_type=brainstate.FakeState)

            # train LoRA weights
            self.lora = brainscale.nn.LoRA(
                in_features=self.pop.in_size,
                lora_rank=n_rank,
                out_features=self.pop.out_size,
                A_init=brainstate.init.LecunNormal(unit=u.mV),
                param_type=conn_param_type
            )

        else:
            raise ValueError('conn_mode must be either "sparse" or "sparse+low+rank"')

    def update(self, x=None):
        """
        Update the network state based on the current input.

        Args:
            x (Optional): External input to the network. Defaults to None.

        Returns:
            dict: A dictionary containing the spike, voltage, and conductance states.
        """
        # Update the input module for the neuron population delayed spikes
        pre_spk = self.delay.at('D')
        pre_spk = jax.lax.stop_gradient(pre_spk)

        # compute recurrent connections and update neurons
        if self.conn_mode == 'sparse+low+rank':
            inp = self.conn(brainevent.EventArray(pre_spk)) * self.scale_factor
            inp = inp + self.lora(pre_spk)
        else:
            raise ValueError('mode must be either "sparse" or "sparse+low+rank')

        if x is None:
            x = inp
        else:
            x += inp
        spk = self.pop(x)

        # update delay spikes
        self.delay.update(jax.lax.stop_gradient(spk))

        return spk


def get_bins(spike_rates, bin_size, neural_activity_max_fr):
    max_firing_rate = spike_rates.max()
    max_firing_rate = np.ceil(max_firing_rate.to_decimal(u.Hz))
    bins = np.arange(0, max_firing_rate, bin_size.to_decimal(u.Hz))
    bins = np.append(bins, neural_activity_max_fr.to_decimal(u.Hz))
    return jnp.asarray(bins)


def neuropil_to_bin_indices(neuropil_fr: u.Quantity[u.Hz], bins):
    """
    Convert neuropil firing rates to bin indices.

    This method maps the given neuropil firing rates to the corresponding bin indices
    based on the pre-defined bins stored in the class instance.

    Args:
        neuropil_fr (u.Quantity[u.Hz]): Firing rates for each neuropil, specified in Hertz units.

    Returns:
        jnp.ndarray: An array of bin indices corresponding to each input firing rate.
    """
    assert bins.ndim == 1, 'bins must be a 1D array'
    bins = bins.to_decimal(u.Hz) if isinstance(bins, u.Quantity) else bins
    neuropil_fr = neuropil_fr.to_decimal(u.Hz) if isinstance(neuropil_fr, u.Quantity) else neuropil_fr

    # Convert the neuropil firing rates to decimal values in Hertz and digitize them
    # based on the pre-defined bins stored in the class instance.
    fn = lambda x: jnp.digitize(x, bins, False)
    for _ in range(neuropil_fr.ndim - 1):
        fn = jax.vmap(fn)
    bin_indices = fn(neuropil_fr)
    return bin_indices


class NeuralActivity(brainstate.nn.Module):
    """
    A class to handle neural activity data for drosophila brain simulations.

    This class manages spike rates data from neural activity recordings and provides
    functionality to map between neuropil-level and neuron-level firing rates.

    Attributes:
        pop_size (brainstate.typing.Size): Size of the neuron population.
        neural_activity_id (str): Identifier for the neural activity dataset.
        conversion (str): Method for converting between neuropil and neuron firing rates ('unique' or 'weighted').
        spike_rates (ndarray): Array of spike rates for different neuropils.
        neuropils (ndarray): Names of the neuropils corresponding to spike_rates.
        neuropil_to_connectivity (NestedDict): Mapping from neuropils to their connectivity information.

    Args:
        pop_size (brainstate.typing.Size): Size of the neuron population.
        flywire_version (str): Version of the flywire dataset to use ('783' or '630').
        neural_activity_id (str, optional): Identifier for the neural activity dataset.
            Defaults to '2017-10-26_1'.
        neural_activity_max_fr (u.Quantity, optional): Maximum firing rate for scaling.
            Defaults to 120 Hz.

    Raises:
        ValueError: If ``flywire_version`` is not one of the supported versions or
                   if conversion is not 'unique' or 'weighted'.
    """

    def __init__(
        self,
        pop: Population,
        flywire_version: str,
        neural_activity_id: str = '2017-10-26_1',
        neural_activity_max_fr: u.Quantity = 120 * u.Hz,
        param_type: type = brainscale.ETraceParam,
        seed: int = 2025,
        bin_size: u.Quantity = 0.1 * u.Hz,
        noise_sigma: float = 0.1,
    ):
        super().__init__()
        self.pop = pop
        self.noise_sigma = noise_sigma

        # uniform binning
        self.rng = np.random.RandomState(seed)
        self.bin_size = bin_size
        print('Loading neural activity information ...')

        # neural activity data
        self.neural_activity_id = neural_activity_id
        data = np.load(os.path.join(base_fn, f'spike_rates/ito_{neural_activity_id}_spike_rate.npz'))
        self.neuropils = data['areas'][1:]
        self.spike_rates = u.math.asarray(data['rates'][1:] * neural_activity_max_fr).T  # [time, neuropil]
        self.bins = get_bins(self.spike_rates, bin_size, neural_activity_max_fr)

        # connectivity data, which show a given neuropil contains which connections
        print('Loading connectivity information ...')
        if flywire_version in ['783', 783]:
            conn_path = os.path.join(base_fn, '783_connections_processed.csv')
        elif flywire_version in ['630', 630]:
            conn_path = os.path.join(base_fn, '630_connections_processed.csv')
        else:
            raise ValueError('flywire_version must be either "783" or "630"')
        connectivity = pd.read_csv(conn_path)
        neuropil_to_connectivity = brainstate.util.NestedDict()
        for i, neuropil in enumerate(self.neuropils):
            # find out all connections (spike source) to a given neuropil
            position = connectivity['neuropil'] == neuropil
            pre_indices = connectivity['pre_index'][position].values
            post_indices = connectivity['post_index'][position].values
            syn_count = connectivity['syn_count'][position].values
            # pre/post-synaptic indices and counts
            (
                pre_indices,
                pre_counts,
                post_indices,
                post_counts,
            ) = count_pre_post_connections(pre_indices, post_indices, syn_count)
            pre_weights = pre_counts / pre_counts.sum()
            post_weights = post_counts / post_counts.sum()
            neuropil_to_connectivity[neuropil] = brainstate.util.NestedDict(
                pre_indices=pre_indices,
                post_indices=post_indices,
                pre_weights=pre_weights,
                post_weights=post_weights,
            )
        self.neuropil_to_connectivity = neuropil_to_connectivity

        # neural activity conversion
        self.neuropil2neuron = brainstate.nn.Sequential(
            brainscale.nn.Linear(
                self.n_neuropil,
                self.pop.varshape,
                w_init=brainstate.init.KaimingNormal(unit=u.mV),
                b_init=brainstate.init.ZeroInit(unit=u.mV),
                param_type=param_type
            ),
            brainscale.nn.ReLU()
        )

    def update(self, embedding):
        noise_weight = self.neuropil2neuron(u.get_mantissa(embedding))

        # excite neurons
        refractory = self.pop.get_refractory()

        # excitation
        brainstate.nn.poisson_input(
            freq=20 * u.Hz,
            num_input=1,
            weight=noise_weight,
            target=self.pop.v,
            refractory=refractory,
        )

    def update_test(self, noise_weight):
        # excite neurons
        refractory = self.pop.get_refractory()

        # excitation
        brainstate.nn.poisson_input(
            freq=20 * u.Hz,
            num_input=1,
            weight=noise_weight,
            target=self.pop.v,
            refractory=refractory,
        )

    @property
    def n_neuropil(self) -> int:
        return self.spike_rates.shape[1]

    @property
    def n_time(self) -> int:
        return self.spike_rates.shape[0]

    @brainstate.compile.jit(static_argnums=0)
    def neuropil_fr_to_embedding(self, neuropil_fr: u.Quantity[u.Hz]):
        """
        Convert firing rates from neuropil-level to embedding-level.
        This method maps firing rates from individual neuropils to the embedding level
        by applying a one-hot encoding to the firing rates.
        
        Args:
            neuropil_fr (u.Quantity[u.Hz]): Firing rates for each neuropil, specified in Hertz units.
            
        Returns:
            u.Quantity[u.Hz]: Embedding-level firing rates, specified in Hertz units.
        """

        def convert(key, fr):
            # convert firing rates to bins
            right_bin_indices = neuropil_to_bin_indices(fr, self.bins)
            left_bin_indices = right_bin_indices - 1
            left = self.bins[left_bin_indices]
            right = self.bins[right_bin_indices]
            return brainstate.random.uniform(left, right, left.shape, key=key)

        if neuropil_fr.ndim == 1:
            return convert(brainstate.random.split_key(), neuropil_fr)
        elif neuropil_fr.ndim == 2:
            return jax.vmap(convert)(brainstate.random.split_key(neuropil_fr.shape[0]), neuropil_fr)
        else:
            raise ValueError

    def neuron_to_neuropil_fr(self, neuron_fr: u.Quantity[u.Hz]):
        """
        Convert firing rates from neuron-level to neuropil-level.

        This method maps firing rates from individual neurons to the neuropil level by
        aggregating the neuron firing rates using weighted sums based on their
        pre-synaptic connections to each neuropil.

        Args:
            neuron_fr (u.Quantity[u.Hz]): Firing rates for each neuron in the population,
                specified in Hertz units.

        Returns:
            u.Quantity[u.Hz]: Firing rates for each neuropil, specified in Hertz units.

        Note:
            The conversion applies the weights defined by the selected conversion method
            ('unique', 'weighted', or 'average') during class initialization.
        """
        neuropil_fr = []
        for i, neuropil in enumerate(self.neuropils):
            # find out all connections (spike source) to a given neuropil
            pre_indices = self.neuropil_to_connectivity[neuropil]['pre_indices']
            pre_weights = self.neuropil_to_connectivity[neuropil]['pre_weights']
            neuropil_fr.append(u.math.sum(neuron_fr[pre_indices] * pre_weights))
        return u.math.asarray(neuropil_fr)

    def read_neuropil_fr(self, i):
        """
        Read the spike rate of a specific neuropil at a given index.

        Args:
            i (int): The index of the spike rate to read from the stored spike rates.

        Returns:
            u.Quantity: The spike rate at the specified index.
        """
        return self.spike_rates[i]

    def iter_data(
        self,
        batch_size: int,
        drop_last: bool = False,
        test_phase: bool = True,
    ):
        """
        Iterate over the neural activity data in batches.

        Args:
            batch_size (int): The size of each batch.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller

        Yields:
            Tuple: A tuple containing the spike rates and neuropils for each batch.
        """
        spike_rates = u.get_mantissa(self.spike_rates)
        # if not test_phase:
        #     spike_rates = np.random.normal(spike_rates, spike_rates * self.noise_sigma)
        #     spike_rates = np.minimum(spike_rates, 0.)
        # plt.imshow(spike_rates.T, aspect='auto')
        # plt.show()


        for i in range(1, self.n_time, batch_size):
            if i + batch_size > self.n_time:
                if drop_last:
                    break
                batch_size = self.n_time - i

            input_neuropil_fr = spike_rates[i - 1:i + batch_size - 1]
            output_neuropil_fr = spike_rates[i:i + batch_size] * u.Hz
            if test_phase:
                input_embed = input_neuropil_fr
            else:
                input_embed = self.neuropil_fr_to_embedding(input_neuropil_fr)
            yield (
                input_embed,
                output_neuropil_fr,
            )


class FiringRateNetwork(brainstate.nn.Module):
    """
    A neural network model for simulating firing rates in the Drosophila brain.

    This class implements a spiking neural network that models the firing rate dynamics
    of neurons in the Drosophila brain. It integrates population dynamics, synaptic
    connections, and neural activity data to provide a comprehensive simulation framework.

    Inherits from brainstate.nn.Module to leverage the neural network functionality.

    Parameters
    ----------
    input_method : str, optional
        Method for processing input to neurons, default is 'relu'.
    flywire_version : str, optional
        Version of the FlyWire connectome to use, default is '630'.
    neural_activity_id : str, optional
        Identifier for the neural activity dataset, default is '2017-10-26_1'.
    neural_activity_max_fr : u.Quantity, optional
        Maximum firing rate for neural activity, default is 100 Hz.
    fr_conversion : str, optional
        Method for converting firing rates between representations, default is 'weighted'.

    Attributes
    ----------
    input_method : str
        Method used for processing neural inputs.
    flywire_version : str
        Version of the FlyWire connectome being used.
    pop_inp : PopulationInput
        Handler for input to the neural population.
    neural_activity : NeuralActivity
        Container for neural activity data and conversion utilities.
    interaction : Interaction
        Neural network for simulating population dynamics.
    """

    def __init__(
        self,
        flywire_version: str = '630',
        neural_activity_id: str = '2017-10-26_1',
        neural_activity_max_fr: u.Quantity = 100. * u.Hz,
        n_rank: int = 20,
        scale_factor=0.3 * 0.275 / 7 * u.mV,
        conn_param_type: type = brainscale.ETraceParam,
        input_param_type: type = brainscale.ETraceParam,
        sampling_rate: u.Quantity = 1.2 * u.Hz,
        seed: int = 2025,
        bin_size: u.Quantity = 0.1 * u.Hz,
        noise_sigma: float = 0.1,
    ):
        super().__init__()

        # parameters
        self.flywire_version = flywire_version
        self.n_sample_step = int(1 / sampling_rate / brainstate.environ.get_dt())

        # population and its input
        self.pop = Population(
            flywire_version,
            V_init=v_init,
            g_init=g_init,
            tau_ref=5.0 * u.ms
        )

        # neural activity data
        self.neural_activity = NeuralActivity(
            pop=self.pop,
            flywire_version=flywire_version,
            neural_activity_id=neural_activity_id,
            neural_activity_max_fr=neural_activity_max_fr,
            param_type=input_param_type,
            seed=seed,
            bin_size=bin_size,
            noise_sigma=noise_sigma,
        )

        # network
        self.interaction = Interaction(
            self.pop,
            n_rank=n_rank,
            scale_factor=scale_factor,
            conn_param_type=conn_param_type,
        )

    def count_neuropil_fr(self, length: int = None):
        if length is None:
            length = self.n_sample_step
        neuron_fr = self.pop.spike_count.value / (length * brainstate.environ.get_dt())
        neuron_fr = neuron_fr.to(u.Hz)
        fun = self.neural_activity.neuron_to_neuropil_fr
        for i in range(neuron_fr.ndim - 1):
            fun = jax.vmap(fun)
        neuropil_fr = fun(neuron_fr)
        return neuropil_fr

    def update(self, i, embedding: u.Quantity):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            # give inputs
            self.neural_activity.update(embedding)

            # update network
            spk = self.interaction.update()
            return spk

    def update_test(self, i, noise_weight):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            # give inputs
            self.neural_activity.update_test(noise_weight)

            # update network
            spk = self.interaction.update()
            return spk

    @brainstate.compile.jit(static_argnums=0)
    def simulate(self, inp_embedding, indices):
        def step_run(i):
            self.update_test(i, noise_weight)

        noise_weight = self.neural_activity.neuropil2neuron(inp_embedding)
        self.pop.reset_spk_count()
        brainstate.compile.for_loop(step_run, indices)
        frs = self.count_neuropil_fr(indices.shape[0])
        return frs


class Trainer:
    """
    A trainer for optimizing neural network models of the Drosophila brain.

    This class handles the training process for firing rate network models, implementing
    optimization strategies to match target neural activity patterns. It sets up the model,
    configures the optimizer, and provides methods for training and evaluation.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimizer, default is 1e-3.
    etrace_decay : float or None, optional
        Decay factor for eligibility traces, default is 0.99. If None, uses parameter-dimension VJP.
    grad_clip : float or None, optional
        Maximum norm for gradient clipping, default is 1.0. If None, no clipping is applied.
    neural_activity_id : str, optional
        Identifier for the neural activity dataset, default is '2017-10-26_1'.
    input_method : str, optional
        Method for processing neural input, default is 'relu'.
    fr_conversion : str, optional
        Method for converting firing rates between representations, default is 'weighted'.
    flywire_version : str, optional
        Version of the FlyWire connectome to use, default is '630'.
    max_firing_rate : u.Quantity, optional
        Maximum firing rate for neural activity, default is 100 Hz.
    sampling_rate : u.Quantity, optional
        Rate at which to sample neural activity, default is 1.2 Hz.

    Attributes
    ----------
    etrace_decay : float or None
        Decay factor for eligibility traces.
    grad_clip : float or None
        Maximum norm for gradient clipping.
    indices : ndarray
        Time indices for simulation and training.
    n_sim : int
        Number of simulation steps before training.
    target : FiringRateNetwork
        The neural network model being trained.
    trainable_weights : dict
        Dictionary of trainable parameters in the model.
    opt : brainstate.optim.Adam
        Optimizer for updating model parameters.
    filepath : str
        Path for saving training results.
    """

    def __init__(
        self,
        sim_before_train: float = 0.5,
        lr: float = 1e-3,
        etrace_decay: float | None = 0.99,
        grad_clip: float | None = 1.0,

        # network parameters
        neural_activity_id: str = '2017-10-26_1',
        flywire_version: str = '630',
        max_firing_rate: u.Quantity = 100. * u.Hz,
        loss_fn: str = 'mse',
        vjp_method: str = 'single-step',
        n_rank: int = 20,
        conn_param_type: type = brainscale.ETraceParam,
        input_param_type: type = brainscale.ETraceParam,
        scale_factor=0.01 * u.mV,
        seed: int = 2025,
        noise_sigma: float = 0.1,
        bin_size: u.Quantity = 0.1 * u.Hz,
    ):
        # parameters
        self.sim_before_train = sim_before_train
        self.etrace_decay = etrace_decay
        self.grad_clip = grad_clip
        self.loss_fn = loss_fn
        self.vjp_method = vjp_method

        # population and its input
        self.target = FiringRateNetwork(
            flywire_version=flywire_version,
            neural_activity_id=neural_activity_id,
            neural_activity_max_fr=max_firing_rate,
            conn_param_type=conn_param_type,
            input_param_type=input_param_type,
            scale_factor=scale_factor,
            n_rank=n_rank,
            seed=seed,
            bin_size=bin_size,
            noise_sigma=noise_sigma,
        )

        # optimizer
        self.trainable_weights = brainstate.graph.states(self.target, brainstate.ParamState)
        self.opt = brainstate.optim.Adam(lr)
        self.opt.register_trainable_weights(self.trainable_weights)

        # train save path
        time_ = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filepath = (
            f'results/'
            f'v4_2/'
            f'{flywire_version}#'
            f'{neural_activity_id}#'
            f'{max_firing_rate / u.Hz}Hz#'
            f'{etrace_decay}#'
            f'{loss_fn}#'
            f'{conn_param_type.__name__}#'
            f'{input_param_type.__name__}#'
            f'{scale_factor.to_decimal(u.mV):5f}#'
            f'{n_rank}#'
            f'{sim_before_train}#'
            f'{seed}#'
            f'{bin_size.to_decimal(u.Hz)}#'
            f'{noise_sigma}#'
            f'{time_}'
        )

    def get_loss(self, current_neuropil_fr, target_neuropil_fr):
        if self.loss_fn == 'mse':
            loss = braintools.metric.squared_error(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'mae':
            loss = braintools.metric.absolute_error(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'huber':
            loss = braintools.metric.huber_loss(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'cosine_distance':
            loss = braintools.metric.cosine_distance(current_neuropil_fr, target_neuropil_fr)
        elif self.loss_fn == 'log_cosh':
            loss = braintools.metric.log_cosh(current_neuropil_fr, target_neuropil_fr)
        else:
            raise ValueError(f'Unknown loss function: {self.loss_fn}')
        return loss

    @brainstate.compile.jit(static_argnums=0)
    def train(self, input_embed, target_neuropil_fr):
        indices = np.arange(self.target.n_sample_step)

        # last_neuropil_fr: [n_batch, n_neuropil]
        # target_neuropil_fr: [n_batch, n_neuropil]
        n_batch = input_embed.shape[0]

        # model
        if self.etrace_decay is None or self.etrace_decay == 0.:
            model = brainscale.ParamDimVjpAlgorithm(self.target, vjp_method=self.vjp_method)
        else:
            model = brainscale.IODimVjpAlgorithm(self.target, self.etrace_decay, vjp_method=self.vjp_method)
        brainstate.nn.vmap_init_all_states(self.target, axis_size=n_batch, state_tag='hidden')

        @brainstate.augment.vmap_new_states(
            state_tag='etrace',
            axis_size=n_batch,
            in_states=self.target.states('hidden')
        )
        def init():
            model.compile_graph(0, input_embed[0])
            model.show_graph()

        init()

        # simulation without record eligibility trace
        n_sim = int(self.sim_before_train * self.target.n_sample_step)
        if n_sim > 0:
            batch_target = brainstate.nn.Vmap(self.target, vmap_states='hidden', in_axes=(None, 0))
            brainstate.compile.for_loop(
                lambda i: batch_target(i, input_embed),
                indices[:n_sim],
            )

        # simulation with eligibility trace recording
        self.target.pop.reset_spk_count(n_batch)
        model = brainstate.nn.Vmap(model, vmap_states=('hidden', 'etrace'), in_axes=(None, 0))
        brainstate.compile.for_loop(
            lambda i: model(i, input_embed),
            indices[n_sim:-1],
        )

        # training
        def loss_fn(i):
            spk = model(i, input_embed)
            current_neuropil_fr = self.target.count_neuropil_fr(self.target.n_sample_step - n_sim)
            loss_ = self.get_loss(current_neuropil_fr, target_neuropil_fr).mean()
            return u.get_mantissa(loss_), current_neuropil_fr

        grads, loss, neuropil_fr = brainstate.augment.grad(
            loss_fn, self.trainable_weights, return_value=True, has_aux=True
        )(indices[-1])
        max_g = jax.tree.map(lambda x: jnp.abs(x).max(), grads)
        if self.grad_clip is not None:
            grads = brainstate.nn.clip_grad_norm(grads, self.grad_clip)
        self.opt.update(grads)

        target_bin_indices = neuropil_to_bin_indices(target_neuropil_fr, self.target.neural_activity.bins)
        predict_bin_indices = neuropil_to_bin_indices(neuropil_fr, self.target.neural_activity.bins)
        acc = jnp.mean(jnp.asarray(target_bin_indices == predict_bin_indices, dtype=jnp.float32))

        return loss, neuropil_fr, max_g, acc

    def show_res(self, neuropil_fr, target_neuropil_fr, i_epoch, i_batch, n_neuropil_per_fig=10):
        fig, gs = braintools.visualize.get_figure(n_neuropil_per_fig, 2, 2., 10)
        for i in range(n_neuropil_per_fig):
            xticks = (i + 1 == n_neuropil_per_fig)
            fig.add_subplot(gs[i, 0])
            barplot(
                self.target.neural_activity.neuropils,
                neuropil_fr[i].to_decimal(u.Hz),
                title='Simulated FR',
                xticks=xticks
            )
            fig.add_subplot(gs[i, 1])
            barplot(
                self.target.neural_activity.neuropils,
                target_neuropil_fr[i].to_decimal(u.Hz),
                title='True FR',
                xticks=xticks
            )
        filename = f'{self.filepath}/images/neuropil_fr-at-epoch-{i_epoch}-batch-{i_batch}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()

    def round1_train(
        self,
        train_epoch: int,
        batch_size: int = 128,
        checkpoint_path: str = None,
    ):
        if checkpoint_path is not None:
            braintools.file.msgpack_load(checkpoint_path, self.target.states(brainstate.ParamState))
            filepath = os.path.join(os.path.dirname(checkpoint_path), 'new')
            os.makedirs(filepath, exist_ok=True)
        else:
            filepath = self.filepath

        # training process
        os.makedirs(filepath, exist_ok=True)
        with open(f'{filepath}/losses.txt', 'w') as file:

            # training
            max_acc = 0.
            for i_epoch in range(train_epoch):
                i_batch = 0
                all_loss = []
                all_acc = []
                for input_embed, target_neuropil_fr in self.target.neural_activity.iter_data(
                    batch_size=batch_size, drop_last=True, test_phase=False
                ):
                    t0 = time.time()
                    res = self.train(input_embed, target_neuropil_fr)
                    loss, neuropil_fr, max_g, acc = jax.block_until_ready(res)
                    t1 = time.time()

                    output(
                        file,
                        f'epoch = {i_epoch}, '
                        f'batch = {i_batch}, '
                        f'loss = {loss:.5f}, '
                        f'bin acc = {acc:.5f}, '
                        f'lr = {self.opt.current_lr:.6f}, '
                        f'time = {t1 - t0:.5f}s'
                    )
                    output(file, f'max_g = {max_g}')

                    i_batch += 1
                    all_loss.append(loss)
                    all_acc.append(acc)
                self.show_res(neuropil_fr, target_neuropil_fr, i_epoch, '')
                self.opt.lr.step_epoch()

                # save checkpoint
                loss = np.mean(all_loss)
                acc = np.mean(all_acc)
                output(
                    file,
                    f'epoch = {i_epoch}, '
                    f'loss = {loss:.5f}, '
                    f'bin acc = {acc:.5f}, '
                    f'lr = {self.opt.current_lr:.6f}'
                )
                if acc > max_acc:
                    braintools.file.msgpack_save(
                        f'{filepath}/best-checkpoint.msgpack',
                        self.target.states(brainstate.ParamState),
                    )
                    max_acc = acc


def load_setting(filepath):
    setting = filepath.split('/')[2].split('#')
    flywire_version = setting[0]
    neural_activity_id = setting[1]
    max_firing_rate = float(setting[2].split('Hz')[0]) * u.Hz
    etrace_decay = eval(setting[3])
    loss_fn = setting[4]
    conn_param_type = setting[5]
    input_param_type = setting[6]
    scale_factor = float(setting[7]) * u.mV
    n_rank = int(setting[8])
    sim_before_train = float(setting[9])
    seed = int(setting[10])
    bin_size = float(setting[11]) * u.Hz
    noise_sigma = float(setting[12])
    return locals()


def first_round_train():
    checkpoint_path = None

    train_epoch = 500
    if checkpoint_path is None:
        neural_activity_id = '2017-10-26_1'
        flywire_version = '630'
        lr, etrace_decay, scale_factor, bin_size, noise_sigma = 1e-2, 0.99, 0.0825 / 100 * u.mV, 0.25 * u.Hz, 0.05
    else:
        lr = brainstate.optim.StepLR(1e-3, step_size=50, gamma=0.9)
        settings = checkpoint_path.split('/')[2].split('#')
        flywire_version = settings[0]
        neural_activity_id = settings[1]
        etrace_decay = float(settings[3])
        conn_param_type = settings[5]
        input_param_type = settings[6]
        scale_factor = float(settings[7]) * u.mV
        n_rank = int(settings[8])
        sim_before_train = float(settings[9])
        bin_size = float(settings[11]) * u.Hz
        noise_sigma = float(settings[12])

    brainstate.environ.set(dt=1.0 * u.ms)
    trainer = Trainer(
        lr=lr,
        etrace_decay=etrace_decay,
        sim_before_train=0.1,
        neural_activity_id=neural_activity_id,
        flywire_version=flywire_version,
        max_firing_rate=100.0 * u.Hz,
        loss_fn='mse',
        scale_factor=scale_factor,
        conn_param_type=brainscale.ETraceParam if etrace_decay != 0. else brainscale.NonTempParam,
        input_param_type=brainscale.ETraceParam if etrace_decay != 0. else brainscale.NonTempParam,
        bin_size=bin_size,
        noise_sigma=noise_sigma,
    )
    trainer.round1_train(
        train_epoch=train_epoch,
        batch_size=128,
        checkpoint_path=checkpoint_path
    )


def generate_training_data():
    brainstate.environ.set(dt=0.2 * u.ms)

    filepath = 'results/v4_2/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#2025#0.25#0.05#2025-04-12-20-55-49/best-checkpoint.msgpack'
    setting = filepath.split('/')[2].split('#')
    flywire_version = setting[0]
    neural_activity_id = setting[1]
    max_firing_rate = float(setting[2].split('Hz')[0]) * u.Hz
    etrace_decay = eval(setting[3])
    loss_fn = setting[4]
    conn_param_type = setting[5]
    input_param_type = setting[6]
    scale_factor = float(setting[7]) * u.mV
    n_rank = int(setting[8])
    sim_before_train = float(setting[9])
    seed = int(setting[10])
    bin_size = float(setting[11]) * u.Hz
    noise_sigma = float(setting[12])

    net = FiringRateNetwork(
        flywire_version=flywire_version,
        neural_activity_id=neural_activity_id,
        neural_activity_max_fr=max_firing_rate,
        conn_param_type=getattr(brainscale, conn_param_type),
        input_param_type=getattr(brainscale, input_param_type),
        n_rank=n_rank,
        scale_factor=scale_factor,
        seed=seed,
        bin_size=bin_size,
        noise_sigma=noise_sigma,
    )
    braintools.file.msgpack_load(filepath, net.states(brainstate.ParamState))
    brainstate.nn.init_all_states(net)

    @brainstate.compile.jit
    def one_step(i, indices):
        input_embed = net.neural_activity.spike_rates[i] / u.Hz
        output_neuropil_fr = net.neural_activity.spike_rates[i + 1]
        net.simulate(input_embed, indices[:n_sim])
        neuropil_fr = net.simulate(input_embed, indices[n_sim:]).to_decimal(u.Hz)
        sim = jnp.corrcoef(u.get_mantissa(output_neuropil_fr), neuropil_fr)[0, 1]
        return neuropil_fr, sim

    n_sim = int(sim_before_train * net.n_sample_step)
    simulated_neuropil_fr = []
    indices = np.arange(net.n_sample_step)
    bar = tqdm(total=net.neural_activity.n_time)
    all_sim = []
    for i in range(0, net.neural_activity.n_time):
        neuropil_fr, sim = one_step(i, indices)
        bar.update()
        bar.set_description(f'similarity = {sim:.5f}')
        indices = indices + net.n_sample_step
        simulated_neuropil_fr.append(neuropil_fr)
        all_sim.append(sim)
    bar.close()
    print(f'Mean similarity = {np.mean(np.mean(all_sim)):.5f}')
    simulated_neuropil_fr = np.asarray(simulated_neuropil_fr)  # [n_time, n_neuropil]
    np.save(os.path.join(os.path.dirname(filepath), 'simulated_neuropil_fr'), simulated_neuropil_fr)


class RNNNet(brainstate.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.norm = brainstate.nn.LayerNorm(n_in, use_scale=False, use_bias=False)
        self.rnn = brainstate.nn.GRUCell(n_in, n_hidden)
        self.readout = brainstate.nn.Linear(n_hidden, n_out)

    def update(self, x):
        norm = self.norm(u.get_mantissa(x))
        rnn = self.rnn(norm)
        readout = self.readout(rnn)
        return u.math.relu(readout) * u.Hz


def second_round_train():
    filepath = 'results/v4_2/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#2025#0.25#0.05#2025-04-12-20-55-49'
    setting = filepath.split('/')[2].split('#')
    flywire_version = setting[0]
    neural_activity_id = setting[1]
    max_firing_rate = float(setting[2].split('Hz')[0]) * u.Hz
    etrace_decay = eval(setting[3])
    # loss_fn = setting[4]
    conn_param_type = setting[5]
    input_param_type = setting[6]
    scale_factor = float(setting[7]) * u.mV
    n_rank = int(setting[8])
    sim_before_train = float(setting[9])
    seed = int(setting[10])
    bin_size = float(setting[11]) * u.Hz
    noise_sigma = float(setting[12])
    noise_sigma = 0.1

    n_hidden = 256
    batch_size = 128
    input_style = 'v1'
    lr = brainstate.optim.StepLR(5e-3, step_size=10, gamma=0.9)
    lr = brainstate.optim.StepLR(1e-3, step_size=10, gamma=0.9)

    data = np.load(os.path.join(base_fn, f'spike_rates/ito_{neural_activity_id}_spike_rate.npz'))
    spike_rates = u.math.asarray(data['rates'][1:] * max_firing_rate).T
    targets = spike_rates[1:]
    bins = get_bins(spike_rates, bin_size, max_firing_rate)
    true_bin_indices = neuropil_to_bin_indices(targets, bins)
    simulated_spike_rates = np.load(os.path.join(filepath, 'simulated_neuropil_fr.npy'))
    # scales = jnp.minimum(simulated_spike_rates * noise_sigma, 0.1)
    scales = 0.1
    # true_scales = spike_rates / u.Hz * noise_sigma
    # true_scales = 0.05

    bin_indices = neuropil_to_bin_indices(spike_rates, bins)
    low_rates = bins[bin_indices - 1]
    high_rates = bins[bin_indices]

    @brainstate.compile.jit
    def generate_inputs():
        simulation_sample_fn = jax.vmap(lambda key: brainstate.random.normal(simulated_spike_rates, scales, key=key))
        simulation_sample_fn = jax.vmap(
            lambda key: jax.random.normal(key, simulated_spike_rates.shape) * scales + simulated_spike_rates)

        if input_style == 'v1':
            # sampled_spike_rates = brainstate.random.normal(spike_rates / u.Hz, true_scales)
            # sampled_spike_rates = jnp.minimum(sampled_spike_rates, 0.)
            # bin_indices = jax.vmap(neuropil_to_bin_indices, in_axes=(0, None))(sampled_spike_rates * u.Hz, bins)
            # low_rates = bins[bin_indices - 1]
            # high_rates = bins[bin_indices]
            true_sample_fn = jax.vmap(lambda key: brainstate.random.uniform(low_rates, high_rates, key=key))
            true_sampling = true_sample_fn(brainstate.random.split_key(batch_size // 2))

            simulation_sampling = simulation_sample_fn(brainstate.random.split_key(batch_size // 2))
            simulation_sampling = jnp.minimum(simulation_sampling, 0.)

            inputs = jnp.concatenate([true_sampling, simulation_sampling], axis=0)
            return jnp.transpose(inputs, (1, 0, 2))

        elif input_style == 'v2':
            simulation_sampling = simulation_sample_fn(brainstate.random.split_key(batch_size))
            simulation_sampling = jnp.minimum(simulation_sampling, 0.)
            return jnp.transpose(simulation_sampling, (1, 0, 2))

        else:
            raise ValueError(f'Unknown input style: {input_style}')

    net = RNNNet(n_in=spike_rates.shape[1], n_hidden=n_hidden, n_out=spike_rates.shape[1])
    weights = net.states(brainstate.ParamState)
    opt = brainstate.optim.Adam(lr=lr)
    opt.register_trainable_weights(weights)

    # braintools.file.msgpack_load(os.path.join(filepath, f'second-round-rnn-checkpoint-{input_style}.msgpack'), weights)

    def f_predict(inputs):
        # inputs = norm(inputs)
        brainstate.nn.vmap_init_all_states(net, axis_size=batch_size)
        return brainstate.compile.for_loop(net, inputs)

    def verify_acc(predictions):
        pred_bin_indices = neuropil_to_bin_indices(predictions, bins)
        acc = jnp.asarray(pred_bin_indices == true_bin_indices, dtype=float).mean()
        return acc

    def loss_fn(predictions):
        mse = (
            u.math.square(u.math.relu(low_rates[1:] - predictions / u.Hz)) +
            u.math.square(u.math.relu(predictions / u.Hz - high_rates[1:]))
        ).mean()
        return mse

    def f_loss(inputs):
        predictions = f_predict(inputs)
        predictions = u.math.transpose(predictions[:-1], (1, 0, 2))
        mse = loss_fn(predictions)
        acc = jax.vmap(verify_acc)(predictions).mean()
        return mse, acc

    @brainstate.compile.jit
    def f_train(inputs):
        grads, l, acc = brainstate.augment.grad(f_loss, weights, return_value=True, has_aux=True)(inputs)
        opt.update(grads)
        return l, acc

    @brainstate.compile.jit
    def f_test():
        brainstate.nn.init_all_states(net)
        outputs = brainstate.compile.for_loop(net, simulated_spike_rates)
        mse = loss_fn(outputs[:-1])
        acc = jax.vmap(verify_acc)(outputs[:-1]).mean()
        return mse, acc

    all_loss = []
    all_acc = []
    min_loss = np.inf
    t0 = time.time()
    for i_epoch in range(1000):
        train_loss_epoch = []
        train_acc_epoch = []
        for i_batch in range(100):
            inputs = generate_inputs()
            loss, acc = f_train(inputs)
            train_acc_epoch.append(acc)
            train_loss_epoch.append(loss)
        acc = np.mean(train_acc_epoch)
        loss = np.mean(train_loss_epoch)
        test_mse, test_acc = f_test()
        print(
            f'Epoch = {i_epoch}, '
            f'Train Loss = {loss:.5f}, '
            f'Train bin acc = {acc:.5f}, '
            f'Test Loss = {test_mse:.5f}, '
            f'Test bin acc = {test_acc:.5f}, '
            f'lr = {opt.current_lr:.6f}, '
            f'time = {time.time() - t0:.2f} s'
        )
        opt.lr.step_epoch()
        if min_loss > test_mse:
            min_loss = test_mse
            braintools.file.msgpack_save(f'{filepath}/second-round-rnn-checkpoint-{input_style}.msgpack',
                                         net.states(brainstate.LongTermState))
        t0 = time.time()

    fig, gs = braintools.visualize.get_figure(1, 2, 3, 4)
    fig.add_subplot(gs[0, 0])
    plt.plot(all_loss)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    fig.add_subplot(gs[0, 1])
    plt.plot(all_acc)
    plt.xlabel('Batch')
    plt.ylabel('Bin accuracy')
    plt.show()


def second_round_loading():
    filepath = 'results/v4_2/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#2025#0.25#0.05#2025-04-12-20-55-49'
    setting = filepath.split('/')[2].split('#')
    flywire_version = setting[0]
    neural_activity_id = setting[1]
    max_firing_rate = float(setting[2].split('Hz')[0]) * u.Hz
    etrace_decay = eval(setting[3])
    # loss_fn = setting[4]
    conn_param_type = setting[5]
    input_param_type = setting[6]
    scale_factor = float(setting[7]) * u.mV
    n_rank = int(setting[8])
    sim_before_train = float(setting[9])
    seed = int(setting[10])
    bin_size = float(setting[11]) * u.Hz
    noise_sigma = float(setting[12])
    noise_sigma = 0.1

    data = np.load(os.path.join(base_fn, f'spike_rates/ito_{neural_activity_id}_spike_rate.npz'))
    spike_rates = u.math.asarray(data['rates'][1:] * max_firing_rate).T
    simulated_spike_rates = np.load(os.path.join(filepath, 'simulated_neuropil_fr.npy'))
    n_hidden = 256

    net = RNNNet(n_in=spike_rates.shape[1], n_hidden=n_hidden, n_out=spike_rates.shape[1])
    norm = brainstate.nn.LayerNorm(spike_rates.shape[1])
    braintools.file.msgpack_load(os.path.join(filepath, f'second-round-rnn-checkpoint-v1.msgpack'),
                                 net.states(brainstate.LongTermState))

    brainstate.nn.init_all_states(net)
    outputs = brainstate.compile.for_loop(lambda x: net(norm(x)), simulated_spike_rates)

    times = np.arange(simulated_spike_rates.shape[0]) * 0.8 * u.second
    num = 5
    for ii in range(0, spike_rates.shape[1], num):
        fig, gs = braintools.visualize.get_figure(num, 2, 4, 8.0)
        for i in range(num):
            # plot simulated neuropil firing rate data
            fig.add_subplot(gs[i, 0])
            data = outputs[:, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
            # plot experimental neuropil firing rate data
            fig.add_subplot(gs[i, 1])
            data = simulated_spike_rates[:, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
        plt.show()


def example_to_simulate():
    import matplotlib.pyplot as plt
    brainstate.environ.set(dt=0.2 * u.ms)
    net = FiringRateNetwork(
        flywire_version='630',
        neural_activity_id='2017-10-26_1',
        neural_activity_max_fr=100 * u.Hz,
        conn_param_type=brainscale.NonTempParam,
        input_param_type=brainscale.NonTempParam,
        n_rank=20,
        # scale_factor=0.0825 / 40 * u.mV,
        scale_factor=0.0825 / 100 * u.mV,
    )
    brainstate.nn.init_all_states(net)

    neuropil_fr = net.neural_activity.read_neuropil_fr(0)
    t0 = 0.0 * u.ms
    t1 = net.n_sample_step * brainstate.environ.get_dt()

    for i in range(10):
        neuropil_fr = net.simulate(neuropil_fr, t0, t0 + t1)
        fig, gs = braintools.visualize.get_figure(2, 1, 3, 10.0)
        fig.add_subplot(gs[0, 0])
        barplot(net.neural_activity.neuropils, neuropil_fr.to_decimal(u.Hz), title='Simulated FR')
        fig.add_subplot(gs[1, 0])
        target_neuropil_fr = net.neural_activity.read_neuropil_fr(i + 1)
        barplot(net.neural_activity.neuropils, target_neuropil_fr.to_decimal(u.Hz), title='True FR')
        plt.show()
        t0 += t1


def example_to_load():
    brainstate.environ.set(dt=0.2 * u.ms)

    # filepath = 'results/v4_2/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#2025#0.5#0.1#2025-04-12-20-54-35'
    filepath = 'results/v4_2/630#2017-10-26_1#100.0Hz#0.99#mse#ETraceParam#ETraceParam#0.000825#20#0.1#2025#0.25#0.05#2025-04-12-20-55-49'
    setting = filepath.split('/')[2].split('#')
    flywire_version = setting[0]
    neural_activity_id = setting[1]
    max_firing_rate = float(setting[2].split('Hz')[0]) * u.Hz
    etrace_decay = eval(setting[3])
    loss_fn = setting[4]
    conn_param_type = setting[5]
    input_param_type = setting[6]
    scale_factor = float(setting[7]) * u.mV
    n_rank = int(setting[8])
    sim_before_train = float(setting[9])
    seed = int(setting[10])
    bin_size = float(setting[11]) * u.Hz
    noise_sigma = float(setting[12])

    # spiking neural network for spike generation
    spiking_net = FiringRateNetwork(
        flywire_version=flywire_version,
        neural_activity_id=neural_activity_id,
        neural_activity_max_fr=max_firing_rate,
        conn_param_type=getattr(brainscale, conn_param_type),
        input_param_type=getattr(brainscale, input_param_type),
        n_rank=n_rank,
        scale_factor=scale_factor,
        seed=seed,
        bin_size=bin_size,
        noise_sigma=noise_sigma,
    )
    braintools.file.msgpack_load(os.path.join(filepath, 'best-checkpoint.msgpack'),
                                 spiking_net.states(brainstate.ParamState))
    brainstate.nn.init_all_states(spiking_net)

    # Recurrent neural network for decoding
    rnn_net = RNNNet(
        n_in=spiking_net.neural_activity.n_neuropil,
        n_hidden=256,
        n_out=spiking_net.neural_activity.n_neuropil
    )
    braintools.file.msgpack_load(os.path.join(filepath, f'second-round-rnn-checkpoint-v1.msgpack'),
                                 rnn_net.states(brainstate.LongTermState))
    brainstate.nn.init_all_states(rnn_net)

    @brainstate.compile.jit
    def process(neuropil_firing_rate, indices):
        rnn_out = rnn_net(neuropil_firing_rate)
        spiking_net.simulate(rnn_out / u.Hz, indices[:n_sim])
        neuropil_fr = spiking_net.simulate(rnn_out / u.Hz, indices[n_sim:])
        target_neuropil_fr = spiking_net.neural_activity.read_neuropil_fr(i + 1)
        target_bin_indices = neuropil_to_bin_indices(target_neuropil_fr, spiking_net.neural_activity.bins)
        predict_bin_indices = neuropil_to_bin_indices(neuropil_fr, spiking_net.neural_activity.bins)
        acc = jnp.mean(jnp.asarray(target_bin_indices == predict_bin_indices, dtype=jnp.float32))
        mse = u.get_mantissa(u.math.square(target_neuropil_fr - neuropil_fr)).mean()
        return neuropil_fr, mse, acc

    # n_time = 200
    n_time = spiking_net.neural_activity.n_time
    n_sim = int(sim_before_train * spiking_net.n_sample_step)
    t1 = spiking_net.n_sample_step * brainstate.environ.get_dt()
    indices = np.arange(spiking_net.n_sample_step)
    bar = tqdm(total=n_time)
    all_accs = []
    all_losses = []
    simulated_neuropil_fr = []
    for i in range(n_time):
        # brainstate.nn.reset_all_states(net)
        if i < 100:
            neuropil_fr = spiking_net.neural_activity.read_neuropil_fr(i)
        neuropil_fr, mse, acc = process(neuropil_fr, indices)
        simulated_neuropil_fr.append(neuropil_fr.to_decimal(u.Hz))
        bar.set_description(f'Bin acc = {acc}, mse = {mse}')
        bar.update(1)
        all_losses.append(float(mse))
        all_accs.append(float(acc))
        indices += spiking_net.n_sample_step
    simulated_neuropil_fr = np.asarray(simulated_neuropil_fr)  # [n_time, n_neuropil]
    print(
        f'Mean bin acc  = {np.mean(all_accs):.5f}, '
        f'mean mse loss = {np.mean(all_losses):.5f}'
    )
    np.save(os.path.join(filepath, 'neuropil_fr_predictions'), simulated_neuropil_fr)

    experimental_neuropil_fr = np.asarray(spiking_net.neural_activity.spike_rates / u.Hz)  # [n_time, n_neuropil]
    times = np.arange(simulated_neuropil_fr.shape[0]) * t1
    num = 5
    for ii in range(0, spiking_net.neural_activity.n_neuropil, num):
        fig, gs = braintools.visualize.get_figure(num, 2, 4, 8.0)
        for i in range(num):
            # plot simulated neuropil firing rate data
            fig.add_subplot(gs[i, 0])
            data = simulated_neuropil_fr[:, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
            # plot experimental neuropil firing rate data
            fig.add_subplot(gs[i, 1])
            data = experimental_neuropil_fr[:n_time, i + ii]
            plt.plot(times, data)
            plt.ylim(0., data.max() * 1.05)
        plt.show()


if __name__ == '__main__':
    first_round_train()
    # generate_training_data()
    # second_round_train()
    # second_round_loading()
    # example_to_load()
    # example_to_simulate()
