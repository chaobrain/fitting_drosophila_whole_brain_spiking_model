# Drosophila Whole Brain Activity Fitting

This project implements a two-stage neural network model to simulate and predict the dynamics of the Drosophila (fruit fly) whole brain activity.


## Data

This project requires two main datasets:

- connectome data from the FlyWire project (version 630/783): https://codex.flywire.ai/
- neural activity recordings from the Drosophila brain: https://doi.org/10.6084/m9.figshare.13349282


We have also provided preprocessed data files in the `data/` directory for convenience.

Please download the datasets (https://drive.google.com/file/d/1TPuHJ-IC1yQtL5TMngAjGJa_JAMdusnu/view?usp=drive_link) and place them in the appropriate directories (`data/`) as specified in the code.



## Key Features

- Loads and processes Drosophila brain connectome data
- Simulates neural activity with biologically plausible dynamics
- Predicts firing rates across brain regions (neuropils)
- Evaluates prediction accuracy using bin classification and MSE metrics
- Generates comprehensive visualizations (heatmaps, time series, correlation plots, error analysis)
- Automatic experiment organization with filepath management
- Smart checkpoint resumption

## Usage

The model follows a complete workflow with 4 stages:

1. **First Round Training (train1)**: Train the spiking neural network to capture brain dynamics
2. **Generate Training Data (eval1)**: Generate training data from the trained spiking network
3. **Second Round Training (train2)**: Train the RNN encoder/decoder to process input signals
4. **Evaluate**: Evaluate the complete model and save predictions with comprehensive visualizations

### Quick Start - Full Pipeline

Run the complete workflow from start to finish:

```bash
python drosophila_whole_brain_fitting.py --mode all --epoch_round1 50 --epoch_round2 50
```

### Running Individual Stages

You can run each stage separately for more control:

```bash
# Stage 1: Train spiking neural network
python drosophila_whole_brain_fitting.py --mode train1 --epoch_round1 500

# Stage 2: Generate training data (requires checkpoint from stage 1)
python drosophila_whole_brain_fitting.py --mode eval1 --filepath results/v4_2/630#2017-10-26_1#...

# Stage 3: Train RNN (requires generated data)
python drosophila_whole_brain_fitting.py --mode train2 --epoch_round2 1000 --filepath results/v4_2/630#2017-10-26_1#...

# Stage 4: Evaluate (requires both checkpoints)
python drosophila_whole_brain_fitting.py --mode evaluate --filepath results/v4_2/630#2017-10-26_1#...
```

### Command Line Arguments

#### Dataset Configuration
- `--flywire_version`: Version of the FlyWire connectome data (`630` or `783`, default: `630`)
- `--neural_activity_id`: ID of the neural activity recording (default: `2017-10-26_1`)
- `--bin_size`: Bin size for discretizing firing rates in Hz (default: `0.25`)
- `--devices`: GPU device IDs, e.g., `"0"` or `"0,1"` (default: `0`)

#### Workflow Control
- `--mode`: Which stage(s) to run (default: `all`)
  - `all`: Run complete pipeline (train1 → eval1 → train2 → evaluate)
  - `train1`: First round training only
  - `eval1`: Generate training data (evaluate stage 1)
  - `train2`: Second round training only
  - `evaluate`: Evaluation only

- `--filepath`: Base directory path for checkpoints and results (default: auto-generated)
  - **If not provided**: Automatically generates a unique directory path based on training parameters
  - **If provided**: Uses the specified directory and loads settings from `first-round-losses.txt`
  - **Examples**:
    - Auto-generated: `results/v4_2/630#2017-10-26_1#100.0Hz#...#2025-12-11-15-30-45`
    - Custom: `results/my_experiment`
  - All checkpoints, logs, and results are saved to this directory
  - Enables easy experiment resumption and organization

#### Training Configuration
- `--epoch_round1`: Number of epochs for first-round training (default: `500`)
- `--epoch_round2`: Number of epochs for second-round training (default: `1000`)
- `--batch_size`: Batch size for training (default: `128`)
- `--lr`: Learning rate for first-round training (default: `0.01`)
- `--lr_round2`: Learning rate for second-round training (default: `0.001`)

#### Model Hyperparameters
- `--etrace_decay`: Decay factor for eligibility traces, `0` for non-temporal (default: `0.99`)
- `--scale_factor`: Scale factor for synaptic connections in mV (default: `0.000825`)
- `--n_rank`: LoRA rank for low-rank adaptation (default: `20`)
- `--n_hidden`: RNN hidden size for second-round training (default: `256`)
- `--sim_before_train`: Fraction of simulation steps before training (default: `0.1`)
- `--noise_sigma`: Noise sigma for data augmentation (default: `0.05`)
- `--max_firing_rate`: Maximum firing rate for neural activity in Hz (default: `100.0`)
- `--loss_fn`: Loss function (`mse`, `mae`, `huber`, `cosine_distance`, `log_cosh`, default: `mse`)
- `--grad_clip`: Gradient clipping value (default: `1.0`)

#### Advanced Options
- `--dt`: Time step for simulation in ms (default: `0.2`)
- `--seed`: Random seed for reproducibility (default: `2025`)
- `--input_style`: Input style for second-round training (`v1` or `v2`, default: `v1`)
- `--split`: Train/test split ratio (informational, default: `0.6`)

### Example Commands

**Quick test run with reduced epochs:**
```bash
python drosophila_whole_brain_fitting.py --mode all --epoch_round1 2 --epoch_round2 2
```

**Full training with custom hyperparameters:**
```bash
python drosophila_whole_brain_fitting.py \
    --mode all \
    --flywire_version 630 \
    --neural_activity_id 2017-10-26_1 \
    --epoch_round1 500 \
    --epoch_round2 1000 \
    --lr 0.01 \
    --batch_size 128 \
    --devices 0
```

**Using custom filepath for experiment organization:**
```bash
python drosophila_whole_brain_fitting.py \
    --mode all \
    --filepath results/my_experiment \
    --epoch_round1 500 \
    --epoch_round2 1000
```

**Resuming from auto-generated filepath:**
```bash
# First run creates auto-generated path
python drosophila_whole_brain_fitting.py --mode train1 --epoch_round1 100

# Resume by providing the generated path
python drosophila_whole_brain_fitting.py \
    --filepath results/v4_2/630#2017-10-26_1#...#2025-12-11-15-30-45 \
    --mode all
```

### Output Files

The workflow creates the following outputs in the results directory:

**Checkpoints:**
- `first-round-checkpoint.msgpack`: Best checkpoint from first-round training (spiking network)
- `second-round-rnn-checkpoint-v1.msgpack`: Best checkpoint from second-round training (RNN)

**Training Data and Logs:**
- `simulated_neuropil_fr.npy`: Generated training data from stage 1 evaluation
- `first-round-losses.txt`: Training logs and hyperparameter settings for first round
- `evaluation_stats.txt`: Evaluation metrics summary (bin accuracy, MSE loss)

**Predictions:**
- `neuropil_fr_predictions.npy`: Final predictions on test data

**Visualizations:**
- `images/`: Training visualizations (neuropil firing rate comparisons during training)
- `evaluation_plots/`: Comprehensive evaluation visualizations including:
  - `heatmap_comparison.png`: Side-by-side heatmaps of ground truth vs simulated activity
  - `time_series_comparison.png`: Temporal evolution for selected neuropils
  - `correlation_scatter.png`: Predicted vs actual firing rates with correlation
  - `barplot_comparison_t*.png`: Bar plot comparisons at 3 sample time points
  - `error_analysis.png`: 4-panel error diagnostics (heatmap, distribution, per-neuropil, over-time)

## Evaluation

The model automatically evaluates performance when running in `all` or `evaluate` mode:

**Metrics:**
- **Bin accuracy**: Percentage of correctly predicted firing rate bins
- **MSE loss**: Mean squared error between predicted and actual firing rates
- **Correlation**: Pearson correlation between predictions and ground truth

**Outputs:**
- Numerical results saved to `evaluation_stats.txt`
- Predictions saved to `neuropil_fr_predictions.npy`
- Comprehensive visualizations saved to `evaluation_plots/` directory

## Visualization

The workflow generates two types of visualizations:

### Training Visualizations (Stage 1)
Generated during first-round training and saved to `images/` directory:
- Neuropil firing rate comparisons (simulated vs ground truth)
- Bar plots for each batch showing model progress

### Evaluation Visualizations (Stage 4)
Generated during final evaluation and saved to `evaluation_plots/` directory:

1. **Heatmap Comparison**: Side-by-side visualization of ground truth and simulated firing rates across all neuropils and time steps
2. **Time Series Plots**: Detailed temporal evolution for 6 neuropils with highest variance
3. **Correlation Scatter Plot**: Overall prediction accuracy with correlation coefficient
4. **Sample Bar Plots**: Neuropil-level comparisons at 25%, 50%, and 75% time points
5. **Error Analysis**: Four-panel comprehensive error diagnostics:
   - Absolute error heatmap across time and neuropils
   - Relative error distribution histogram
   - Mean error per neuropil bar plot
   - Mean error over time line plot

All visualizations are publication-ready (150 DPI, proper labels and legends).


[//]: # (## Citation )

[//]: # ()
[//]: # ()
[//]: # (If you use this code or data, please cite:)

[//]: # ()
[//]: # (```text)

[//]: # (@article {Wang2024.09.24.614728,)

[//]: # (	author = {Wang, Chaoming and Dong, Xingsi and Ji, Zilong and Jiang, Jiedong and Liu, Xiao and Wu, Si},)

[//]: # (	title = {BrainTrace: Enabling Scalable Online Learning in Spiking Neural Networks},)

[//]: # (	elocation-id = {2024.09.24.614728},)

[//]: # (	year = {2025},)

[//]: # (	doi = {10.1101/2024.09.24.614728},)

[//]: # (	publisher = {Cold Spring Harbor Laboratory},)

[//]: # (	URL = {https://www.biorxiv.org/content/early/2025/07/27/2024.09.24.614728},)

[//]: # (	eprint = {https://www.biorxiv.org/content/early/2025/07/27/2024.09.24.614728.full.pdf},)

[//]: # (	journal = {bioRxiv})

[//]: # (})

[//]: # (```)
