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

import datetime
import functools
import os
import time
from pathlib import Path
from typing import Callable

import brainevent
import brainscale
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy import signal, optimize

from setting import Setting

dftype = brainstate.environ.dftype()

data_path = './data'


def v_init(shape):
    """
    Initialize membrane voltage values for neurons with random normal distribution.

    This function creates an array of initial membrane voltages using a random normal
    distribution with mean 0.0 and standard deviation 0.8. The values are scaled to
    millivolts using the brainunit library.

    Parameters
    ----------
    shape : tuple
        The shape of the output array, defining the dimensions of the neuron population.

    Returns
    -------
    brainstate.random.RandomState.normal : u.Quantity
        An array of initialized membrane voltages with the specified shape, drawn from
        a normal distribution and scaled to millivolts.
    """
    # return brainstate.random.RandomState(42).normal(0., 0.8, shape) * u.mV
    return jnp.zeros(shape) * u.mV


def g_init(shape):
    """
    Initialize synaptic conductance values for neurons with random uniform distribution.

    This function creates an array of initial synaptic conductances using a random uniform
    distribution between 0.0 and 0.2. The values are scaled to millivolts using the
    brainunit library.

    Parameters
    ----------
    shape : tuple
        The shape of the output array, defining the dimensions of the neuron population.

    Returns
    -------
    brainstate.random.RandomState.uniform : u.Quantity
        An array of initialized synaptic conductances with the specified shape, drawn from
        a uniform distribution and scaled to millivolts.
    """
    # return brainstate.random.RandomState(2025).uniform(0., 0.2, shape) * u.mV
    return jnp.zeros(shape) * u.mV


def output(file, msg: str):
    """
    Write a message to both the console and a file.

    This function outputs the given message to the standard output (console) and
    writes it to the specified file, appending a newline character. The file is
    flushed immediately after writing to ensure data is committed to disk.

    Parameters
    ----------
    file : file-like object
        An open file-like object with write and flush methods. Typically this is
        a file opened in write or append mode.
    msg : str
        The message string to be output to console and written to the file.

    Returns
    -------
    None
        This function does not return any value.

    Examples
    --------
    >>> with open('log.txt', 'w') as f:
    ...     output(f, 'Processing started')
    Processing started
    """
    print(msg)
    file.write(msg + '\n')
    file.flush()


def df_f_to_firing_rate(
    df_f,
    tau_rise: float = 0.07,
    tau_decay: float = 0.5,
    dt: float = 0.033,
    method: str = 'deconvolution'
):
    """
    Convert calcium imaging ΔF/F signals into estimated neuronal firing rates.

    Parameters:
    -----------
    df_f : array-like
        The ΔF/F calcium signal time series.
    tau_rise : float, optional
        Rise time constant of calcium transient in seconds. Default is 0.07s.
    tau_decay : float, optional
        Decay time constant of calcium transient in seconds. Default is 0.5s.
    dt : float, optional
        Time step in seconds. Default is 0.033s (30 Hz).
    method : str, optional
        Method for conversion:
        - 'deconvolution': Wiener deconvolution (default)
        - 'derivative': Simple derivative method
        - 'foopsi': Fast non-negative deconvolution

    Returns:
    --------
    firing_rate : ndarray
        Estimated neuronal firing rate in events/second.
    """
    df_f = np.array(df_f)

    if method == 'derivative':
        # Simple derivative method
        # Taking derivative and rectifying
        d_df_f = np.diff(df_f, prepend=df_f[0])
        firing_rate = np.maximum(0, d_df_f) / dt

    elif method == 'deconvolution':
        # Wiener deconvolution method
        # Create calcium impulse response function
        t = np.arange(0, 5 * tau_decay, dt)
        h = (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
        h = h / np.sum(h * dt)  # Normalize

        # Perform deconvolution in frequency domain
        n = len(df_f)
        # Zero padding
        df_f_padded = np.hstack((df_f, np.zeros(len(h))))
        h_padded = np.hstack((h, np.zeros(n)))

        # FFT
        df_f_fft = np.fft.fft(df_f_padded)
        h_fft = np.fft.fft(h_padded)

        # Signal-to-noise ratio estimation (SNR)
        # This is a simplified approach; ideally, estimate from data
        snr = 10  # Example SNR value
        noise_power = np.mean(np.abs(df_f) ** 2) / snr

        # Wiener deconvolution
        c = 1 / (h_fft + noise_power / (np.abs(h_fft) ** 2))
        firing_rate_fft = df_f_fft * c
        firing_rate_full = np.real(np.fft.ifft(firing_rate_fft))

        # Take only the relevant part and rectify
        firing_rate = np.maximum(0, firing_rate_full[:n])

    elif method == 'foopsi':
        # Fast non-negative deconvolution (simplified version)
        g = np.exp(-dt / tau_decay)  # Discrete time constant

        # Objective function for optimization
        @numba.njit
        def objective(s):
            c = np.zeros_like(s)
            for t in range(1, len(s)):
                c[t] = g * c[t - 1] + s[t]
            return np.sum((c - df_f) ** 2) + 0.5 * np.sum(s)  # L1 regularization

        # Constraints: sparsity and non-negativity
        bounds = [(0, None) for _ in range(len(df_f))]

        # Initial guess
        s0 = np.zeros_like(df_f)

        # Optimization
        result = optimize.minimize(objective, s0, bounds=bounds, method='L-BFGS-B')
        firing_rate = result.x / dt

    else:
        raise ValueError(f"Method '{method}' not recognized. Use 'derivative', 'deconvolution', or 'foopsi'.")

    return firing_rate


@numba.njit
def _create_decay_matrix(frame_count, tau_frames):
    i_indices = np.arange(frame_count)[:, np.newaxis]
    j_indices = np.arange(frame_count)[np.newaxis, :]
    # Create matrix where element (i,j) contains (i-j) value
    time_diffs = i_indices - j_indices
    # Apply exponential decay only where i >= j (time_diffs >= 0)
    decay_matrix = np.where(time_diffs >= 0, np.exp(-time_diffs / tau_frames), 0)
    return decay_matrix


def deconvolve_dff_to_spikes(
    dff,
    tau: float = 0.8 * u.second,
    sampling_rate: float = 30. * u.Hz,
    lambda_reg: float = 0.01,
    smooth: bool = True
):
    """
    Convert df/f calcium signals to estimated spike rates using
    a simplified deconvolution approach.

    Parameters:
    -----------
    dff : numpy.ndarray
        The df/f calcium signal time series
    tau : float
        Decay time constant of calcium transient (in seconds)
    sampling_rate : float
        Sampling rate of the signal (Hz)
    lambda_reg : float
        Regularization parameter for sparsity
    smooth : bool
        Whether to apply smoothing to the df/f signal

    Returns:
    --------
    firing_rate : numpy.ndarray
        Estimated firing rate in Hz
    """
    # Convert tau to units of frames
    tau_frames = float(u.maybe_decimal(tau * sampling_rate))

    # Optional smoothing of df/f signal
    if smooth:
        window_size = min(21, len(dff) // 3)
        # Make window_size odd
        if window_size % 2 == 0:
            window_size += 1
        dff = signal.savgol_filter(dff, window_size, 2)

    # Create calcium decay matrix (for deconvolution)
    frame_count = len(dff)
    decay_matrix = _create_decay_matrix(frame_count, tau_frames)

    # Convert to sparse matrix for computational efficiency
    # decay_matrix = sparse.csr_matrix(decay_matrix)

    # Use non-negative least squares with regularization to solve for firing rates
    # min ||decay_matrix * spikes - dff||^2 + lambda * ||spikes||_1
    @numba.njit
    def objective(spikes):
        predicted_dff = decay_matrix.dot(spikes)
        error = np.sum((predicted_dff - dff) ** 2) + lambda_reg * np.sum(np.abs(spikes))
        return error

    # Initial guess of zero spikes
    x0 = np.zeros_like(dff)

    # Non-negative bound constraint
    bounds = [(0, None) for _ in range(frame_count)]

    # Minimize the objective function
    result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    spikes = result.x

    # Convert to firing rate in Hz (spikes per second)
    firing_rate = spikes * sampling_rate

    return firing_rate


def deconvolve_dff_to_spikes_v2(
    dff,
    tau: float = 0.8 * u.second,
    sampling_rate: float = 30. * u.Hz,
    lambda_reg: float = 0.01,
    smooth: bool = True
):
    """
    Convert df/f calcium signals to estimated spike rates using
    a simplified deconvolution approach.

    Parameters:
    -----------
    dff : numpy.ndarray
        The df/f calcium signal time series
    tau : float
        Decay time constant of calcium transient (in seconds)
    sampling_rate : float
        Sampling rate of the signal (Hz)
    lambda_reg : float
        Regularization parameter for sparsity
    smooth : bool
        Whether to apply smoothing to the df/f signal

    Returns:
    --------
    firing_rate : numpy.ndarray
        Estimated firing rate in Hz
    """
    # Convert tau to units of frames
    tau_frames = float(u.maybe_decimal(tau * sampling_rate))

    # Optional smoothing of df/f signal
    if smooth:
        window_size = min(21, len(dff) // 3)
        # Make window_size odd
        if window_size % 2 == 0:
            window_size += 1
        dff = signal.savgol_filter(dff, window_size, 2)

    # Create calcium decay matrix (for deconvolution)
    frame_count = len(dff)
    decay_matrix = _create_decay_matrix(frame_count, tau_frames)

    # Convert to sparse matrix for computational efficiency
    # decay_matrix = sparse.csr_matrix(decay_matrix)

    # Use non-negative least squares with regularization to solve for firing rates
    # min ||decay_matrix * spikes - dff||^2 + lambda * ||spikes||_1
    @numba.njit
    def objective(spikes):
        predicted_dff = decay_matrix.dot(spikes)
        error = np.sum((predicted_dff - dff) ** 2) + lambda_reg * np.sum(np.abs(spikes))
        return error

    # Initial guess of zero spikes
    x0 = np.zeros_like(dff)

    # Non-negative bound constraint
    bounds = [(0, None) for _ in range(frame_count)]

    # Minimize the objective function
    result = optimize.minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    spikes = result.x

    # Convert to firing rate in Hz (spikes per second)
    firing_rate = spikes * sampling_rate

    return firing_rate


def filter_region_response(region_response, cutoff=None, fs=None):
    """
    Filter a neural region response trace with a high-pass Butterworth filter.

    This function applies a high-pass Butterworth filter to the input neural region
    response data when both cutoff frequency and sampling rate are provided. If either
    parameter is missing, the original signal is returned unfiltered.

    Parameters
    ----------
    region_response : numpy.ndarray
        Neural region response trace data to be filtered.
    cutoff : float, optional
        High-pass filter cutoff frequency in Hz. If None, no filtering is applied.
    fs : float, optional
        Sampling frequency of the signal in Hz. If None, no filtering is applied.

    Returns
    -------
    numpy.ndarray
        Filtered (or original if fs is None) region response data.

    Notes
    -----
    The function uses a first-order high-pass Butterworth filter from scipy.signal
    with second-order sections (SOS) implementation for numerical stability.

    Despite the function name suggesting low-pass filtering, it actually
    implements a high-pass filter ('hp') as specified in the filter design.
    """
    if fs is not None:
        sos = signal.butter(1, cutoff, 'hp', fs=fs, output='sos')
        region_response_filtered = signal.sosfilt(sos, region_response)
    else:
        region_response_filtered = region_response

    return region_response_filtered


def trim_region_response(
    file_id,
    region_response,
    start_include=100,
    end_include=None
):
    """
    Trim artifacts from neural recording data by excluding problematic time periods.

    This function handles artifact removal in neural recordings by either:
    1. Using predefined trimming indices for known problematic recordings, or
    2. Applying default trimming based on start and end indices

    The function handles both multi-ROI recordings and single-dimension behavioral data.

    Parameters
    ----------
    file_id : str
        Identifier for the recording file. Used to match against known problematic recordings.
    region_response : numpy.ndarray
        Neural activity data with one of these shapes:
        - (n_rois, n_frames): Multi-ROI neural activity recordings
        - (n_frames,): Single-dimension binary behavioral response
    start_include : int, optional
        Index of first frame to include when using default trimming. Defaults to 100.
    end_include : int or None, optional
        Index of last frame to include when using default trimming. If None,
        includes all frames after start_include. Defaults to None.

    Returns
    -------
    numpy.ndarray
        Trimmed version of the input data with problematic periods removed.
        Maintains the same dimensionality as the input.

    Raises
    ------
    ValueError
        If region_response doesn't have either 1 or 2 dimensions.

    Notes
    -----
    The function maintains a dictionary of known problematic recordings with custom
    trimming indices to handle specific artifacts like dropouts or baseline shifts.
    """

    # Key: brain file id
    # Val: time inds to include
    # dropout halfway through
    brains_to_trim = {
        '2018-10-19_1': np.array(list(range(100, 900)) + list(range(1100, 2000))),  # transient dropout spikes
        '2017-11-08_1': np.array(list(range(100, 1900)) + list(range(2000, 4000))),  # baseline shift
        '2018-10-20_1': np.array(list(range(100, 1000)))
    }

    if file_id in brains_to_trim.keys():
        include_inds = brains_to_trim[file_id]
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, include_inds]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[include_inds]
        else:
            raise ValueError

    else:  # use default start / end
        if len(region_response.shape) == 2:
            region_response_trimmed = region_response[:, start_include:end_include]
        elif len(region_response.shape) == 1:
            region_response_trimmed = region_response[start_include:end_include]
        else:
            raise ValueError
    return region_response_trimmed


def load_setting(filepath: str | Path) -> Setting:
    with open(os.path.join(filepath, 'training-losses.txt'), 'r') as f:
        lines = eval(f.readline())
    return Setting(lines)


def show_difference(old_setting: Setting, new_setting: Setting) -> dict:
    difference = dict()
    old_setting = old_setting.to_dict()
    new_setting = new_setting.to_dict()
    for key in old_setting:
        if key not in new_setting:
            difference[key] = None
        elif old_setting[key] != new_setting[key]:
            difference[key] = new_setting[key]
    return difference


def new_filepath(checkpoint_path, settings):
    filepath = os.path.dirname(checkpoint_path)
    diff = show_difference(load_setting(filepath), settings)
    if len(diff) > 0:
        filepath = os.path.join(filepath, '-'.join([f'{k}={v}' for k, v in diff.items()]))
    return filepath


def load_syn(flywire_version: str | int, scale_factor: float) -> brainevent.CSR:
    """
    Load synaptic connectivity data from the FlyWire connectome dataset.

    This function loads the neuronal connection data for the Drosophila brain from a
    specified FlyWire connectome version and constructs a Compressed Sparse Row (CSR)
    matrix representation of the synaptic connectivity.

    Parameters
    ----------
    flywire_version : str or int
        Version identifier for the FlyWire connectome dataset.
        Accepted values are '630', 630, '783', or 783.
    scale_factor: float
        The scaling factor to apply to the synaptic weights, typically used to adjust
        the overall strength of the connections.

    Returns
    -------
    brainevent.CSR
        A compressed sparse row matrix representing the synaptic connectivity,
        where each entry (i,j) represents the connection weight from neuron i to neuron j.

    Raises
    ------
    ValueError
        If the flywire_version is not one of the supported versions ('630', 630, '783', 783).

    Notes
    -----
    The function processes the connectivity data by:
    1. Loading neuron information to determine the total number of neurons
    2. Loading synaptic connections from the parquet file
    3. Sorting connections by presynaptic neuron indices
    4. Converting the data to CSR format
    """
    if flywire_version in ['783', 783]:
        path_neu = os.path.join(data_path, 'Completeness_783.csv')
        path_syn = os.path.join(data_path, 'Connectivity_783.parquet')
    elif flywire_version in ['630', 630]:
        path_neu = os.path.join(data_path, 'Completeness_630_final.csv')
        path_syn = os.path.join(data_path, 'Connectivity_630_final.parquet')
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

    weight = jnp.asarray(weight, dtype=dftype) * scale_factor
    csr = brainevent.CSR((weight, indices, indptr), shape=(n_neuron, n_neuron))
    return csr


def split_train_test(
    length: int,
    split: float,
    batch_size: int = None
):
    """
    Split a dataset into training and testing sets based on specified proportions and batch size.

    This function divides the total dataset length into training and testing segments,
    ensuring the training set size is compatible with the specified batch size.

    Parameters
    ----------
    length : int
        Total number of samples in the dataset.
    split : float
        Proportion of data to use for training (0.0 to 1.0).
    batch_size : int
        Size of batches for training. The training set size will be adjusted to be
        divisible by this value (plus 1 sample).

    Returns
    -------
    tuple[int, int]
        A tuple containing (n_train, n_test) where:
        - n_train: Number of samples in the training set
        - n_test: Number of samples in the testing set

    Notes
    -----
    The function ensures that the training set size is compatible with batching by:
    1. Calculating the raw split based on the specified proportion
    2. Adjusting to be divisible by batch_size
    3. Adding 1 to ensure there's always at least one training sample
    """
    if batch_size is None:
        batch_size = 1
    n_train = int(length * split) // batch_size * batch_size + 1
    n_test = length - n_train
    return n_train, n_test


@numba.njit
def count_pre_post_connections(
    pre_indices,
    post_indices,
    syn_counts,
):
    """
    Counts the total number of synaptic connections for each pre- and post-synaptic neuron.

    This function iterates through the provided pre-synaptic indices, post-synaptic indices,
    and their corresponding synaptic counts to determine the total number of connections
    associated with each neuron. It aggregates the synaptic counts for each pre- and
    post-synaptic neuron using dictionaries, then converts the results into NumPy arrays
    for efficient computation and use in other functions.

    Args:
        pre_indices (numpy.ndarray): A 1D NumPy array containing the indices of pre-synaptic neurons.
        post_indices (numpy.ndarray): A 1D NumPy array containing the indices of post-synaptic neurons.
        syn_counts (numpy.ndarray): A 1D NumPy array containing the number of synaptic connections
            between corresponding pre- and post-synaptic neurons.

    Returns:
        tuple: A tuple containing four NumPy arrays:
            - pre_indices (numpy.ndarray): A 1D NumPy array of unique pre-synaptic neuron indices.
            - pre_counts (numpy.ndarray): A 1D NumPy array of the total synaptic counts for each
              pre-synaptic neuron, corresponding to the indices in `pre_indices`.
            - post_indices (numpy.ndarray): A 1D NumPy array of unique post-synaptic neuron indices.
            - post_counts (numpy.ndarray): A 1D NumPy array of the total synaptic counts for each
              post-synaptic neuron, corresponding to the indices in `post_indices`.
    """
    pre_to_count = {}
    post_to_count = {}
    for pre, post, syn_count in zip(pre_indices, post_indices, syn_counts):
        if pre not in pre_to_count:
            pre_to_count[pre] = 0
        pre_to_count[pre] += syn_count

        if post not in post_to_count:
            post_to_count[post] = 0
        post_to_count[post] += syn_count

    pre_indices = []
    pre_counts = []
    for pre, count in pre_to_count.items():
        pre_indices.append(pre)
        pre_counts.append(count)

    post_indices = []
    post_counts = []
    for post, count in post_to_count.items():
        post_indices.append(post)
        post_counts.append(count)
    return np.asarray(pre_indices), np.asarray(pre_counts), np.asarray(post_indices), np.asarray(post_counts)


class NeuralData:
    """
    A class for handling neural activity data from the Drosophila brain.

    This class manages neural firing rate data, connectivity information,
    and provides methods for data conversion between neuron and neuropil levels.
    It supports data loading, preprocessing, and iteration for training neural networks.

    Parameters
    ----------
    flywire_version : str
        Version identifier for the FlyWire connectome dataset ('630' or '783').
    neural_activity_id : str
        Identifier for the neural activity dataset to load.
    split : float, optional
        Proportion of data to use for training vs. testing, defaults to 0.5.
    neural_activity_max_fr : u.Quantity, optional
        Maximum firing rate for neural activity, defaults to 120 Hz.
    duration_per_data : u.Quantity, optional
        Duration of each data sample, defaults to 20.0 ms.
    duration_per_fr : u.Quantity, optional
        Duration for firing rate calculation, defaults to 20.0 ms.
    n_fr_per_warmup : int, optional
        Number of firing rate samples for warmup phase, defaults to 10.
    n_fr_per_train : int, optional
        Number of firing rate samples for training phase, defaults to 90.
    interpolation : str, optional
        Method for interpolating firing rates, defaults to 'linear'.
    noise_sigma : float, optional
        Standard deviation of noise added to firing rates, defaults to 0.0.

    Attributes
    ----------
    neuropils : numpy.ndarray
        Names of neuropils (brain regions) in the dataset.
    spike_rates : u.Quantity
        Neural firing rate data with shape [time, neuropil].
    neuropil_to_connectivity : brainstate.util.NestedDict
        Dictionary mapping neuropil names to connectivity information.
    n_train : int
        Number of samples in the training dataset.
    n_test : int
        Number of samples in the testing dataset.

    Methods
    -------
    n_neuropil
        Returns the number of neuropils in the dataset.
    n_time
        Returns the number of time points in the dataset.
    count_neuropil_fr(neuron_fr)
        Converts firing rates from neuron-level to neuropil-level.
    train_data
        Returns the training portion of the spike rate data.
    test_data
        Returns the testing portion of the spike rate data.
    iter_train_data(batch_size)
        Iterates over the training data in batches.
    iter_test_data(batch_size)
        Iterates over the testing data in batches.
    """

    def __init__(
        self,
        flywire_version: str,
        neural_activity_id: str,
        split: float = 0.5,
        neural_activity_max_fr: u.Quantity = 120 * u.Hz,
        duration_per_data: u.Quantity = 20.0 * u.ms,
        duration_per_fr: u.Quantity = 20.0 * u.ms,
        n_fr_per_warmup: int = 10,
        n_fr_per_train: int = 90,
        n_fr_per_gap: int = 100,
        interpolation: str = 'linear',
        noise_sigma: float = 0.
    ):
        # neural activity data
        self.neural_activity_id = neural_activity_id
        data = np.load(os.path.join(data_path, f'spike_rates/ito_{neural_activity_id}_spike_rate.npz'))
        self.neuropils = data['areas'][1:]
        self.duration_per_data = duration_per_data
        self.duration_per_fr = duration_per_fr
        self.n_train_per_data = int(duration_per_data / duration_per_fr)
        self.n_step_per_grad = int(duration_per_fr / brainstate.environ.get_dt())
        self.n_fr_per_warmup = n_fr_per_warmup
        self.n_fr_per_train = n_fr_per_train
        self.n_fr_per_gap = n_fr_per_gap
        self.interpolation = interpolation
        self.noise_sigma = noise_sigma

        self.spike_rates = u.math.asarray(data['rates'][1:] * neural_activity_max_fr).T  # [time, neuropil]
        xs = jnp.arange(self.spike_rates.shape[0]) * duration_per_data
        xvals = jnp.arange(self.spike_rates.shape[0] * self.n_train_per_data) * duration_per_fr
        fn = lambda ys: u.math.interp(xvals / u.ms, xs / u.ms, ys / u.Hz)
        self.spike_rates = np.asarray(jax.vmap(fn, in_axes=1, out_axes=1)(self.spike_rates)) * u.Hz

        # connectivity data, which show a given neuropil contains which connections
        print('Loading connectivity information ...')
        if flywire_version in ['783', 783]:
            conn_path = os.path.join(data_path, '783_connections_processed.csv')
        elif flywire_version in ['630', 630]:
            conn_path = os.path.join(data_path, '630_connections_processed.csv')
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
            (pre_indices, pre_counts, post_indices, post_counts) = (
                count_pre_post_connections(pre_indices, post_indices, syn_count))
            pre_weights = pre_counts / pre_counts.sum()
            post_weights = post_counts / post_counts.sum()
            neuropil_to_connectivity[neuropil] = brainstate.util.NestedDict(
                pre_indices=pre_indices, post_indices=post_indices,
                pre_weights=pre_weights, post_weights=post_weights,
            )
        self.neuropil_to_connectivity = neuropil_to_connectivity

        # training/testing data split
        self.split = split
        self.n_train, self.n_test = split_train_test(self.n_time, split)

    @property
    def n_neuropil(self) -> int:
        """
        Get the number of neuropils in the dataset.

        Returns:
            int: Number of neuropils.
        """
        return self.spike_rates.shape[1]

    @property
    def n_time(self) -> int:
        """
        Get the number of time points in the dataset.

        Returns:
            int: Number of time points.
        """
        return self.spike_rates.shape[0]

    def _neuron_to_neuropil_fr(self, neuron_fr: u.Quantity[u.Hz]):
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

    def count_neuropil_fr(self, neuron_fr: u.Quantity):
        """
        Convert spike counts to firing rates at the neuropil level.

        This method calculates firing rates from spike counts and maps them to
        neuropil-level firing rates using the connectivity information.

        Args:
            neuron_fr: u.Quantity. Firing rates for each neuron in the population, specified in Hz units.

        Returns:
            u.Quantity: Firing rates for each neuropil, in Hz units.
        """
        neuron_fr = neuron_fr.to(u.Hz)
        fun = self._neuron_to_neuropil_fr
        for i in range(neuron_fr.ndim - 1):
            fun = jax.vmap(fun)
        neuropil_fr = fun(neuron_fr)
        return neuropil_fr

    @property
    def train_data(self):
        """
        Get the training portion of the spike rate data.

        Returns:
            u.Quantity: Training data with shape [n_train, n_neuropil].
        """
        return self.spike_rates[:self.n_train]

    @property
    def test_data(self):
        """
        Get the testing portion of the spike rate data.

        Returns:
            u.Quantity: Testing data with shape [n_test, n_neuropil].
        """
        return self.spike_rates[self.n_train:]

    def iter_train_data_v1(self, batch_size: int = 128, n_batch: int = 100):
        """
        Iterate over the neural activity training data in batches.

        Provides batches of data for training, where each batch contains input firing rates
        and their corresponding target output firing rates for the next time step.

        Yields:
            Tuple[u.Quantity, u.Quantity]: A tuple containing:
                - input_neuropil_fr: Input firing rates transformed to embeddings
                - output_neuropil_fr: Target output firing rates for the next time step
        """
        assert self.noise_sigma > 0., "Noise sigma must be greater than 0 to add noise to the data."
        data = self.spike_rates.mantissa[: self.n_fr_per_warmup]
        target = self.spike_rates.mantissa[1:self.n_fr_per_warmup + 1]
        i_start = self.n_fr_per_warmup + 1
        predict = self.spike_rates.mantissa[i_start: i_start + self.n_fr_per_train]
        targets = np.expand_dims(target, axis=1) * u.Hz
        predicts = np.expand_dims(predict, axis=1) * u.Hz

        for i in range(n_batch):
            inputs = np.maximum(np.random.normal(data, data * self.noise_sigma, (batch_size,) + data.shape), 0.)
            inputs = np.transpose(np.asarray(inputs), (1, 0, 2)) * u.Hz
            yield inputs, targets, predicts

    def iter_train_data_v2(self, batch_size: int = 128):
        """
        Iterate over the neural activity training data in batches.

        Provides batches of data for training, where each batch contains input firing rates
        and their corresponding target output firing rates for the next time step.

        Yields:
            Tuple[u.Quantity, u.Quantity]: A tuple containing:
                - input_neuropil_fr: Input firing rates transformed to embeddings
                - output_neuropil_fr: Target output firing rates for the next time step
        """

        print('Number of training data:', self.n_train)
        data = self.spike_rates.mantissa
        if self.noise_sigma > 0.:
            input_data = np.maximum(np.random.normal(data, data * self.noise_sigma), 0.)
        else:
            input_data = data

        inputs, targets, predicts = [], [], []
        for i in range(0, self.n_train - self.n_fr_per_warmup - self.n_fr_per_train, self.n_fr_per_gap):
            i1 = i + self.n_fr_per_warmup
            i3 = i1 + self.n_fr_per_train
            inputs.append(input_data[i: i1])
            targets.append(data[i + 1: i1 + 1])
            predicts.append(data[i1 + 1: i3])
            if len(inputs) == batch_size:
                inputs = np.transpose(np.asarray(inputs), (1, 0, 2)) * u.Hz
                targets = np.transpose(np.asarray(targets), (1, 0, 2)) * u.Hz
                predicts = np.transpose(np.asarray(predicts), (1, 0, 2)) * u.Hz
                yield inputs, targets, predicts
                inputs, targets, predicts = [], [], []
        if len(inputs) > 0:
            inputs = np.transpose(np.asarray(inputs), (1, 0, 2)) * u.Hz
            targets = np.transpose(np.asarray(targets), (1, 0, 2)) * u.Hz
            predicts = np.transpose(np.asarray(predicts), (1, 0, 2)) * u.Hz
            yield inputs, targets, predicts


class Population(brainstate.nn.Neuron):
    """
    A population of neurons with leaky integrate-and-fire dynamics.

    This class implements a population of leaky integrate-and-fire neurons for the Drosophila brain
    simulation, with connectivity based on the FlyWire connectome dataset. Each neuron follows
    standard LIF dynamics with customizable parameters for membrane properties, synaptic
    transmission, and spike generation.

    The dynamics of the neurons are given by the following equations::

       dv/dt = (v_0 - v + g) / t_mbr  :  volt (unless refractory)
       dg/dt = -g / tau               :  volt (unless refractory)

    Parameters
    ----------
    flywire_version : str or int, optional
        Version of the FlyWire connectome dataset to use ('630' or '783'), defaults to '783'
    v_rest : u.Quantity, optional
        The resting potential of the neurons, defaults to 0 mV
    v_reset : u.Quantity, optional
        The reset potential of the neurons after a spike, defaults to 0 mV
    v_th : u.Quantity, optional
        The threshold potential of the neurons for spiking, defaults to 1 mV
    tau_ref : u.Quantity, optional
        The refractory period of the neurons, defaults to 2.2 ms
    spk_fun : Callable, optional
        The spike function of the neurons, defaults to ReluGrad with width=1.5
    name : str, optional
        The name of the population
    """

    def __init__(
        self,
        flywire_version: str | int = '783',
        v_rest: u.Quantity = 0 * u.mV,  # resting potential
        v_reset: u.Quantity = 0 * u.mV,  # reset potential after spike
        v_th: u.Quantity = 1 * u.mV,  # potential threshold for spiking
        tau_ref: u.Quantity | None = 2.2 * u.ms,  # refractory period
        spk_fun: Callable = brainstate.surrogate.ReluGrad(width=1.5),  # spike function
        name: str = None,
    ):
        # connectome data
        if flywire_version in ['783', 783]:
            path_neu = os.path.join(data_path, 'Completeness_783.csv')
            path_syn = os.path.join(data_path, 'Connectivity_783.parquet')
        elif flywire_version in ['630', 630]:
            path_neu = os.path.join(data_path, 'Completeness_630_final.csv')
            path_syn = os.path.join(data_path, 'Connectivity_630_final.parquet')
        else:
            raise ValueError('flywire_version must be either "783" or "630"')
        self.flywire_version = flywire_version

        # file paths
        self.path_neu = Path(path_neu)
        self.path_syn = Path(path_syn)

        print('Loading neuron information ...')

        # neuron ids
        flywire_ids = pd.read_csv(self.path_neu, index_col=0)
        self.n_neuron = len(flywire_ids)

        super().__init__(self.n_neuron, name=name)

        # parameters
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_th = v_th
        dt = brainstate.environ.get_dt() / u.ms
        self.tau_m_lim = [np.exp(-dt / 1), np.exp(-dt / 200)]
        self.tan_syn_lim = [np.exp(-dt / 1), np.exp(-dt / 200)]
        self.tau_m = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.tau_m_lim[0], self.tau_m_lim[1], self.varshape)
        )
        self.tau_syn = brainscale.ElemWiseParam(
            brainstate.random.uniform(self.tan_syn_lim[0], self.tan_syn_lim[1], self.varshape)
        )

        self.tau_ref = tau_ref  # if tau_ref is None else u.math.full(self.varshape, tau_ref)
        self.spk_fun = spk_fun

    def init_state(self):
        self.v = brainscale.ETraceState(brainstate.init.param(v_init, self.varshape))
        self.g = brainscale.ETraceState(brainstate.init.param(g_init, self.varshape))
        self.spike_count = brainscale.ETraceState(jnp.zeros(self.varshape))
        if self.tau_ref is not None:
            self.t_ref = brainstate.HiddenState(
                brainstate.init.param(brainstate.init.Constant(-1e7 * u.ms), self.varshape)
            )

    def count_firing_rate(self, duration: u.Quantity[u.ms]):
        return self.spike_count.value / duration.to(u.second)

    def reset_firing_rate(self):
        self.spike_count.value = jnp.zeros_like(self.spike_count.value)

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
        neu_decay = jnp.clip(self.tau_m.execute(), min=self.tau_m_lim[0], max=self.tau_m_lim[1])
        syn_decay = jnp.clip(self.tau_syn.execute(), min=self.tan_syn_lim[0], max=self.tan_syn_lim[1])

        # numerical integration + external input current
        g = syn_decay * self.g.value + x
        v = neu_decay * self.v.value + (1 - neu_decay) * (self.v_rest + g)  # synaptic filtering
        v = self.sum_delta_inputs(v)

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


class SparseLinear(brainstate.nn.Module):
    def __init__(
        self,
        sparse_mat: u.sparse.SparseMatrix,
        param_type: type = brainscale.ETraceParam,
        weight_mask: int = 0,
    ):
        super().__init__()

        assert isinstance(sparse_mat, u.sparse.SparseMatrix), '"weight" must be a brainunit.sparse.SparseMatrix.'

        denominator = np.sum(sparse_mat.shape) / 2
        stddev = (jnp.sqrt(2.0 / denominator) / .87962566103423978) * 0.02
        weight = brainstate.random.truncated_normal(-2, 2, sparse_mat.data.shape) * stddev
        params = dict(weight=weight.flatten())  # only train non-zero weights
        if weight_mask == 1:
            self.sign = np.where(u.get_mantissa(sparse_mat.data) > 0, 1, -1)
            op = brainscale.SpMatMulOp(sparse_mat=sparse_mat, weight_fn=self.weight_fn)  # x @ sparse matrix
        else:
            op = brainscale.SpMatMulOp(sparse_mat=sparse_mat)  # x @ sparse matrix
        self.weight_op = param_type(params, op=op)

    def weight_fn(self, w):
        return u.math.abs(w) * self.sign

    def update(self, x):
        return self.weight_op.execute(x)


class InputEncoder(brainstate.nn.Module):
    """
    A module for encoding background inputs to a neural population.

    This class transforms embedding inputs into appropriate noise weights
    for background stimulation of neurons in a population. It provides a mechanism
    for introducing controlled background activity in a spiking neural network.

    The encoder uses a neural network to transform input embeddings into
    weights that are then used to modulate Poisson noise inputs to the neurons.
    This allows for realistic background activity simulation in the neural population.

    Parameters
    ----------
    n_in : int
        Number of input features in the embedding vector.
    n_rec : int
        Number of the recurrent units.
    """

    def __init__(self, n_in: int, n_rec: int):
        super().__init__()

        # neural activity conversion
        self.encoder = brainstate.nn.Sequential(
            brainstate.nn.LayerNorm(n_in, use_scale=False, use_bias=False),
            brainscale.nn.Linear(
                n_in, n_rec,
                w_init=brainstate.init.KaimingNormal(scale=0.1),
                b_init=brainstate.init.ZeroInit(),
            ),
            brainscale.nn.ReLU(),
            lambda x: x * u.mV
        )

    def update(self, firing_rate):
        res = self.encoder(u.get_mantissa(firing_rate))
        return res


class RecurrentNetwork(brainstate.nn.Module):
    def __init__(
        self,
        data: NeuralData,
        n_neuropil: int,
        flywire_version: str,
        n_rank: int = 20,
        tau_ref: float = 5.0,
        weight_mask: int = 0,
        fitting_target: str = 'csr',
        input_method: str = 'current',
    ):
        super().__init__()

        self.data = data
        self.fitting_target = fitting_target
        self.input_method = input_method
        self.n_neuropil = n_neuropil

        # neuronal dynamics
        self.pop = Population(flywire_version, tau_ref=tau_ref if tau_ref is None else tau_ref * u.ms)

        # delay for changes in post-synaptic neuron
        # Paul et al 2015 doi: https://doi.org/10.3389/fncel.2015.00029
        self.delay = brainstate.nn.Delay(
            jax.ShapeDtypeStruct(self.pop.varshape, brainstate.environ.dftype()), entries={'D': 1.8 * u.ms}
        )

        # synaptic dynamics
        csr = load_syn(self.pop.flywire_version, 0.01)
        if fitting_target == 'lora':
            self.conn = brainstate.nn.SparseLinear(csr, b_init=None, param_type=brainstate.FakeState)
            self.lora = brainscale.nn.LoRA(
                in_features=self.pop.in_size,
                lora_rank=n_rank,
                out_features=self.pop.out_size,
                A_init=brainstate.init.LecunNormal(),
                param_type=brainscale.ETraceParam,
            )

        elif fitting_target == 'csr':
            self.conn = SparseLinear(csr, weight_mask=weight_mask)

        else:
            raise ValueError('fitting_target must be either "lora" or "csr"')

        # inputs
        self.input_encoder = InputEncoder(n_neuropil, self.pop.varshape)

    def update(self, i, background_inputs):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            # background inputs to the population
            if self.input_method == 'noise':
                brainstate.nn.poisson_input(freq=20 * u.Hz, num_input=1, weight=background_inputs,
                                            target=self.pop.v, refractory=self.pop.get_refractory())
            elif self.input_method == 'current':
                self.pop.add_delta_input('external-current', background_inputs)
            else:
                raise ValueError('unknown method, must be either "current" or "noise"')

            # recurrent connections
            pre_spk = self.delay.at('D')
            pre_spk = jax.lax.stop_gradient(pre_spk)
            inp = self.conn(brainevent.EventArray(pre_spk))
            if self.fitting_target == 'lora':
                inp = inp + self.lora(pre_spk)

            # update population dynamics
            spk = self.pop(inp * u.mV)

            # update delay spikes
            self.delay.update(jax.lax.stop_gradient(spk))
            return spk


class DrosophilaSpikingNetworkTrainer:
    def __init__(self, settings: Setting):
        # parameters
        self.settings = settings
        self.grad_clip = 1.0

        # inputs
        self.data = NeuralData(
            flywire_version=settings.flywire_version,
            neural_activity_id=settings.neural_activity_id,
            split=settings.split,
            neural_activity_max_fr=settings.neural_activity_max_fr * u.Hz,
            duration_per_data=settings.duration_per_data * u.ms,
            duration_per_fr=settings.duration_per_fr * u.ms,
            n_fr_per_warmup=settings.n_fr_per_warmup,
            n_fr_per_train=settings.n_fr_per_train,
            noise_sigma=settings.input_noise_sigma,
        )

        # population
        self.target = RecurrentNetwork(
            data=self.data,
            n_neuropil=self.data.n_neuropil,
            flywire_version=settings.flywire_version,
            n_rank=settings.n_lora_rank,
            tau_ref=settings.tau_ref,
            fitting_target=settings.fitting_target,
            weight_mask=settings.weight_mask,
        )

        # optimizer
        self.trainable_weights = brainstate.graph.states(self.target, brainstate.ParamState)
        # self.opt = brainstate.optim.Adam(brainstate.optim.StepLR(settings.lr, step_size=50, gamma=0.9))
        self.opt = brainstate.optim.Adam(settings.lr)
        self.opt.register_trainable_weights(self.trainable_weights)

        # train save path
        self.filepath = f"{settings.to_filepath()}#{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    def _loss(self, predict_fr, target_fr):
        if self.data.noise_sigma > 0.:
            lowers = target_fr * (1 - self.data.noise_sigma)
            uppers = target_fr * (1 + self.data.noise_sigma)
            return u.get_mantissa(
                u.math.mean(
                    u.math.square(u.math.relu(lowers - predict_fr)) +
                    u.math.square(u.math.relu(predict_fr - uppers))
                )
            )
        else:
            return u.get_mantissa(u.math.abs(predict_fr - target_fr).mean())

    def _train_target_phase(self, model_and_etrace, carray, inputs):
        grads, prediction = carray
        i_run, input_activity, target_activity = inputs
        indices = np.arange(self.data.n_step_per_grad) + i_run * self.data.n_step_per_grad
        # simulation with eligibility trace recording
        self.target.pop.reset_firing_rate()
        bg_neuron_inputs = self.target.input_encoder(input_activity)
        brainstate.compile.for_loop(lambda i: model_and_etrace(i, bg_neuron_inputs), indices[:-1])

        def loss_fn():
            bg_neuron_inputs_ = self.target.input_encoder(input_activity)
            model_and_etrace(indices[-1], bg_neuron_inputs_)
            neuron_fr = self.target.pop.count_firing_rate(self.data.duration_per_fr)
            predict_neuropil_fr_ = self.data.count_neuropil_fr(neuron_fr)
            loss_ = self._loss(predict_neuropil_fr_, target_activity)
            return loss_, predict_neuropil_fr_

        new_grads, loss, predict_neuropil_fr = brainstate.augment.grad(
            loss_fn, self.trainable_weights, return_value=True, has_aux=True)()
        return (jax.tree.map(jnp.add, grads, new_grads), predict_neuropil_fr), loss

    def _train_predict_phase(self, model, carry, inputs):
        grads, input_neuropil_fr = carry
        batch_index, target_activity = inputs
        indices = np.arange(self.data.n_step_per_grad) + batch_index * self.data.n_step_per_grad

        # simulation with eligibility trace recording
        self.target.pop.reset_firing_rate()
        bg_neuron_inputs = self.target.input_encoder(input_neuropil_fr)
        brainstate.compile.for_loop(lambda i: model(i, bg_neuron_inputs), indices[:-1])

        # gradients and optimizations
        def loss_fn():
            bg_neuron_inputs_ = self.target.input_encoder(input_neuropil_fr)
            model(indices[-1], bg_neuron_inputs_)
            neuron_fr = self.target.pop.count_firing_rate(self.data.duration_per_fr)
            predict_neuropil_fr_ = self.data.count_neuropil_fr(neuron_fr)
            loss_ = self._loss(predict_neuropil_fr_, target_activity)
            return loss_, predict_neuropil_fr_

        new_grads, loss, predict_neuropil_fr = brainstate.augment.grad(
            loss_fn, self.trainable_weights, return_value=True, has_aux=True)()
        return (jax.tree.map(jnp.add, grads, new_grads), predict_neuropil_fr), loss

    @brainstate.compile.jit(static_argnums=0)
    def batch_train(self, input_neuropil_fr, target_neuropil_fr, predict_neuropil_fr):
        # input_neuropil_fr: [n_seq, n_batch, n_neuropil]
        # target_neuropil_fr: [n_seq, n_batch, n_neuropil]
        batch_size = input_neuropil_fr.shape[1]

        # model
        if self.settings.etrace_decay == 0.:
            print('Using forward gradient backpropagation with D-RTRL.')
            model_and_etrace = brainscale.ParamDimVjpAlgorithm(self.target, vjp_method=self.settings.vjp_method)
        else:
            print('Using forward gradient backpropagation with ES-D-RTRL.')
            model_and_etrace = brainscale.IODimVjpAlgorithm(
                self.target, self.settings.etrace_decay, vjp_method=self.settings.vjp_method
            )

        brainstate.nn.vmap_init_all_states(self.target, state_tag='hidden', axis_size=batch_size)

        @brainstate.augment.vmap_new_states(
            state_tag='etrace', axis_size=batch_size, in_states=self.target.states('hidden')
        )
        def init():
            model_and_etrace.compile_graph(0, jnp.zeros(self.target.pop.varshape, dtype=dftype) * u.mV)
            model_and_etrace.show_graph()

        init()
        model_and_etrace = brainstate.nn.Vmap(model_and_etrace, vmap_states=('hidden', 'etrace'), in_axes=(None, 0))

        grads = jax.tree.map(lambda x: jnp.zeros_like(x), self.trainable_weights.to_dict_values())

        # whole-brain network warmup
        (grads, prediction), loss_phase1 = brainstate.transform.scan(
            functools.partial(self._train_target_phase, model_and_etrace),
            (grads, input_neuropil_fr[0]),
            (np.arange(input_neuropil_fr.shape[0]), input_neuropil_fr, target_neuropil_fr)
        )

        # prediction and training
        (grads, prediction), loss_phase2 = brainstate.transform.scan(
            functools.partial(self._train_predict_phase, model_and_etrace),
            (grads, prediction),
            (np.arange(predict_neuropil_fr.shape[0]) + input_neuropil_fr.shape[0], predict_neuropil_fr),
        )

        # gradient clipping and update
        max_g = jax.tree.map(lambda x: jnp.abs(x).max(), grads)
        if self.grad_clip is not None:
            grads = brainstate.functional.clip_grad_norm(grads, self.grad_clip)
        self.opt.update(grads)

        return loss_phase1.mean() + loss_phase2.mean(), max_g

    def f_train(self, train_epoch: int, checkpoint_path: str = None):
        if checkpoint_path is not None:
            braintools.file.msgpack_load(checkpoint_path, self.target.states(brainstate.ParamState))
            filepath = new_filepath(checkpoint_path, self.settings)
        else:
            filepath = self.filepath

        # training process
        os.makedirs(filepath, exist_ok=True)
        file = open(f'{filepath}/training-losses.txt', 'w')
        try:
            output(file, str(self.settings))
            min_loss = np.inf
            for i_epoch in range(train_epoch):
                # training
                i_batch = 0
                all_loss = []
                # for data in self.data.iter_train_data_v2(batch_size=self.settings.batch_size):
                for data in self.data.iter_train_data_v1(batch_size=self.settings.batch_size, n_batch=10):
                    t0 = time.time()
                    loss, max_g = jax.block_until_ready(self.batch_train(*data))
                    t1 = time.time()
                    output(file, f'epoch={i_epoch}, train batch = {i_batch}, loss = {loss:.5f}, time = {t1 - t0:.5f}s')
                    output(file, f'max_g = {max_g}')
                    all_loss.append(loss)
                    i_batch += 1
                self.opt.lr.step_epoch()
                train_loss = np.mean(all_loss)
                lr = self.opt.lr()
                output(file, f'epoch = {i_epoch}, train loss = {train_loss:.5f}, lr = {lr:.6f}')

                # save checkpoint
                if train_loss < min_loss:
                    braintools.file.msgpack_save(
                        f'{filepath}/checkpoint-best-loss={train_loss:.4f}.msgpack',
                        self.trainable_weights,
                    )
                    min_loss = train_loss
        finally:
            file.close()

    @brainstate.compile.jit(static_argnums=0)
    def batch_eval(self, input_neuropil_fr, target_neuropil_fr, predict_neuropil_fr):
        batch_size = input_neuropil_fr.shape[1]
        brainstate.nn.vmap_init_all_states(self.target, state_tag='hidden', axis_size=batch_size)

        # simulation
        self.target.pop.reset_firing_rate()
        model = brainstate.nn.Vmap(self.target, vmap_states='hidden', in_axes=(None, 0))

        def simulation(i_run, input_activity):
            indices = np.arange(self.data.n_step_per_grad) + i_run * self.data.n_step_per_grad
            # simulation with eligibility trace recording
            self.target.pop.reset_firing_rate()
            bg_neuron_inputs = self.target.input_encoder(input_activity)
            brainstate.compile.for_loop(lambda i: model(i, bg_neuron_inputs), indices)
            neuron_fr = self.target.pop.count_firing_rate(self.data.duration_per_fr)
            predict_neuropil_fr_ = self.data.count_neuropil_fr(neuron_fr)
            return predict_neuropil_fr_

        neuropil_fr_phase1 = brainstate.transform.for_loop(
            simulation, np.arange(input_neuropil_fr.shape[0]), input_neuropil_fr
        )
        loss_phase1 = self._loss(neuropil_fr_phase1, target_neuropil_fr)

        def simulation_and_loss(neuro_fr, i_run):
            predict_neuropil_fr_ = simulation(i_run, neuro_fr)
            return predict_neuropil_fr_, predict_neuropil_fr_

        # whole-brain network warmup
        _, neuropil_fr_phase2 = brainstate.transform.scan(
            simulation_and_loss,
            neuropil_fr_phase1[-1],
            np.arange(predict_neuropil_fr.shape[0]) + input_neuropil_fr.shape[0]
        )
        loss_phase2 = self._loss(neuropil_fr_phase2, predict_neuropil_fr)
        return loss_phase1 + loss_phase2, u.math.concatenate([neuropil_fr_phase1, neuropil_fr_phase2], axis=0)

    def f_eval(self, checkpoint_path: str):
        if checkpoint_path is not None:
            braintools.file.msgpack_load(checkpoint_path, self.trainable_weights)
            filepath = os.path.join(os.path.dirname(checkpoint_path), 'prediction')
        else:
            filepath = self.filepath
        os.makedirs(filepath, exist_ok=True)

        # testing
        i_batch = 0
        # for data in self.data.iter_train_data_v2(batch_size=self.settings.batch_size):
        for data in self.data.iter_train_data_v1(batch_size=4, n_batch=1):
            loss, predictions = jax.block_until_ready(self.batch_eval(*data))
            input_neuropil_fr, target_neuropil_fr, predict_neuropil_fr = data
            truths = u.math.concatenate([target_neuropil_fr, predict_neuropil_fr], axis=0)
            for i in range(predictions.shape[1]):
                self.visualize(predictions[:, i], truths[:, i], filename=f'{filepath}/test_batch_{i + i_batch}.png')
            i_batch += predictions.shape[1]
            print(f'Test batch = {i_batch}, loss = {loss}')

    def visualize(self, predictions, groundtruth, filename: str = None):
        # [n_seq, n_neuropil]
        assert predictions.shape == groundtruth.shape, f'{predictions.shape} != {groundtruth.shape}'
        n_neuropil = predictions.shape[1]
        fig, gs = braintools.visualize.get_figure(n_neuropil, 1, 1, 20)
        for i in range(n_neuropil):
            fig.add_subplot(gs[i, 0])
            plt.plot(predictions[:, i], label='Predicted', color='blue')
            plt.plot(groundtruth[:, i], label='Ground Truth', color='orange')
            plt.legend()
        if filename:
            print('Saving figure to', filename)
            plt.savefig(filename, dpi=300)
        plt.close()
