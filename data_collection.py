import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter
from hmm_discretization import Discrete_Model_Optimization
from scipy.stats import qmc
from torch.distributions import MultivariateNormal
import random


# Unused, replaced by score function in hmmlearn
def eval_estimates(n_components, obs, m, c):
    dists = [MultivariateNormal(torch.tensor(m[i]), torch.tensor(c[i]))
             for i in range(n_components)]
    log_prob_vals = [
        dists[i].log_prob(obs).detach().numpy() for i in range(n_components)]
    avg = np.maximum(*log_prob_vals).mean()
    return avg


def generate_parameters(n_dim, n_hidden):
    """Generates random parameters in a range for testing the
    Discr. optimization

    Args:
        n_dim (int): Number of dimensions
        n_hidden (int): Number of hidden states

    Returns:
        np.arrays: means, cov. matrix, startprob and transmat
    """
    means_options = np.linspace(-10.0, 10.0, 51)
    covars_options = np.linspace(0.00, 2.0, 21)
    means_sel = random.choices(means_options, k=n_dim * n_hidden)
    means_sel = np.array(means_sel).reshape((n_hidden, n_dim))
    covars_sel = random.choices(covars_options, k=n_dim * n_dim * n_hidden)
    covars_sel = np.array(covars_sel).reshape((n_hidden, n_dim, n_dim))
    t_covars_sel = torch.tril(torch.tensor(covars_sel))
    t_covars_sel = t_covars_sel @ torch.transpose(t_covars_sel, dim0=1, dim1=2)
    covars_sel = t_covars_sel.detach().numpy()
    if n_hidden == 2:
        startprob = np.array([0.5, 0.5])
        transmat = np.array([[0.8, 0.2], [0.7, 0.3]])
    elif n_hidden == 3:
        startprob = np.array([0.5, 0.2, 0.3])
        transmat = np.array(
            [[0.7, 0.2, 0.1], [0.3, 0.1, 0.6], [0.2, 0.6, 0.2]])
    return means_sel, covars_sel, startprob, transmat


def grid_vals(mod_sel):
    """Function to order the number of times the Discrete observation
        is selected
    Args:
        mod_sel (hmm Discrete model): Discretized model utilized
        to  optimize parameters

    Returns:
        np.arrays: Discrete observations, sizes, discrete sequence,
        observations
    """
    counts = Counter(mod_sel.knn_indices)
    for i in range(mod_sel.sample_size):
        counts[i] += 1
    ord_counts = dict(sorted(counts.items(), key=lambda x: x[0]))
    sizes = np.array(list(ord_counts.values()))
    sizes[np.where(sizes <= 1)] = 1
    sizes[np.where((sizes > 1) & (sizes <= 100))] = 32
    sizes[np.where((sizes > 100) & (sizes <= 500))] = 64
    sizes[np.where((sizes > 500) & (sizes <= 10000))] = 256
    sizes[np.where(sizes > 10000)] = 1024
    discrete_obs = mod_sel.get_discrete_obs()
    return discrete_obs, sizes, mod_sel.discrete_seq, mod_sel.observations


def plot_grids3D(grid_size, num_obs, mods_saved):
    """Plot the 3D grids showing the discrete points, the continuous
    gaussian observations and highlighting the most utilized discrete
    points.

    Args:
        grid_size (int): Size of the grid
        num_obs (int): Number of observations
        mods_saved (hmm Discrete model): Discretized model utilized
        to  optimize parameters
    """
    disc_methods = ['Random', 'Grid', 'LH', 'Sobol', 'Halton']
    fig = plt.figure(figsize=(18, 5))
    title_text = f"Parameters Optimization with 2^{grid_size} discrete" \
        f"points and Sample size = {num_obs}"
    fig.suptitle(title_text)
    for i in range(5):
        ax = fig.add_subplot(1, 5, i+1, projection='3d')
        ax.set_title(f"Disc Method: {disc_methods[i]}")
        disc_obs, sizes, mod_seq, mod_obs = grid_vals(mods_saved[i])
        ax.scatter(mod_seq[:, 0],
                   mod_seq[:, 1],
                   mod_seq[:, 2],
                   color='gray',
                   alpha=0.3,
                   marker='+')
        ax.scatter(mod_obs[:, 0],
                   mod_obs[:, 1],
                   mod_obs[:, 2],
                   color='blue',
                   alpha=0.3,
                   marker='*',
                   label='Continuous')
        ax.scatter(disc_obs[:, 0],
                   disc_obs[:, 1],
                   disc_obs[:, 2],
                   color='red',
                   alpha=0.5,
                   s=sizes,
                   label='Discrete')
        ax.set_zticklabels([])
        ax.set_zticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])
    plt.tight_layout()
    plt.show()


def plot_grids2D(grid_size, num_obs, Q_emps, Q_ests, Q_ests_start, mods_saved):
    """Plot the 2D grids showing the discrete points, the continuous
    gaussian observations and highlighting the most utilized discrete
    points. Also, plots the difference between different Q matrices

    Args:
        grid_size (int): Size of the grid
        num_obs (int): Number of observations
        Q_emps (np.array): Q empirical matrix
        Q_ests (np.array): Q final estimated matrix
        Q_ests_start (np.array): Q initial estimated matrix
        mods_saved (hmm Discrete model): Discretized model utilized
        to  optimize parameters
    """
    disc_methods = ['Random', 'Grid', 'LH', 'Sobol', 'Halton']
    fig, axes = plt.subplots(ncols=5, nrows=4, figsize=(15, 12))
    title_text = f"Parameters Optimization with 2^{grid_size} discrete" \
        f" points and Sample size = {num_obs}"
    fig.suptitle(title_text, fontsize=14)
    for i in range(5):
        axes[0, i].imshow(Q_ests_start[i], aspect='auto')
        axes[0, i].set_title(f'Method: {disc_methods[i]}')
        axes[1, i].imshow(Q_ests[i], aspect='auto')
        axes[2, i].imshow(Q_emps[i], aspect='auto')
        disc_obs, sizes, mod_seq, mod_obs = grid_vals(mods_saved[i])
        axes[3, i].scatter(mod_seq[:, 0],
                           mod_seq[:, 1],
                           color='gray',
                           alpha=0.3,
                           marker='+')
        axes[3, i].scatter(mod_obs[:, 0],
                           mod_obs[:, 1],
                           color='blue',
                           alpha=0.3,
                           marker='*',
                           label='Continuous')
        axes[3, i].scatter(disc_obs[:, 0],
                           disc_obs[:, 1],
                           color='red',
                           alpha=0.5,
                           s=sizes,
                           label='Discrete')
    for ax in axes.flatten():
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])
    axes[0, 0].set_ylabel('Initial estimated Q Matrix', fontsize=12)
    axes[1, 0].set_ylabel('Optimal estimated Q Matrix', fontsize=12)
    axes[2, 0].set_ylabel('Empirical Q Matrix', fontsize=12)
    axes[3, 0].set_ylabel('Visual Discretization', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_metrics(scores,
                 accs,
                 discrepancies,
                 grid_size,
                 num_obs):
    disc_methods = ['Random', 'Grid', 'LH', 'Sobol', 'Halton']
    loss_functions = ['MSE', 'KLD']
    _, axes = plt.subplots(ncols=3, figsize=(15, 5))
    axes[0].plot(scores[:5], label=loss_functions[0])
    axes[0].plot(scores[5:], label=loss_functions[1])
    axes[0].set_xticks(np.arange(len(disc_methods)), disc_methods)
    ax_title = 'Scores for Discretization methods.' \
        f' Size={num_obs}. Grid=2^{grid_size}'
    axes[0].set_title(ax_title, fontsize=10)
    axes[0].set_xlabel('Methods')
    axes[0].set_ylabel('Log-probability')
    axes[0].legend()
    axes[1].plot(accs[:5], label=loss_functions[0])
    axes[1].plot(accs[5:], label=loss_functions[1])
    axes[1].set_xticks(np.arange(len(disc_methods)), disc_methods)
    ax_title = 'Accuracy of Discretization ' \
        f'methods. Size={num_obs}. Grid=2^{grid_size}'
    axes[1].set_title(
        ax_title, fontsize=10)
    axes[1].set_xlabel('Methods')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[2].plot(discrepancies[:5], label=loss_functions[0])
    axes[2].plot(discrepancies[5:], label=loss_functions[1])
    axes[2].set_xticks(np.arange(len(disc_methods)), disc_methods)
    ax_title = 'Discrepancy of Discretization ' \
        f'methods. Size={num_obs}. Grid=2^{grid_size}'
    axes[2].set_title(ax_title, fontsize=8)
    axes[2].set_xlabel('Methods')
    axes[2].set_ylabel('Discrepancy')
    axes[2].legend()
    plt.tight_layout()
    plt.show()


def plot_loss(losses_all, grid_size, num_obs):
    """Plot the average loss of the diferent discretization techniques with
    different losses.

    Args:
        losses_all (list): List of losses to plot
        grid_size (int): Size of the grid
        num_obs (int): Number of observations

    Returns:
        list: average of the loss through the epochs
    """
    disc_methods = ['Random', 'Grid', 'LH', 'Sobol', 'Halton']
    avg_to_plot = list()
    for j in range(len(losses_all)):
        avg_vals = [sum(i) / len(i) for i in zip(*losses_all[j])]
        avg_to_plot.append(avg_vals)
    _, axes = plt.subplots(ncols=2, figsize=(15, 5))
    for c, i in enumerate(disc_methods):
        axes[0].plot(avg_to_plot[c], label=i)
        axes[1].plot(avg_to_plot[c+5], label=i)
        axes[0].legend()
        axes[1].legend()
    axes[0].set_title(f"MSE Loss. Size={num_obs}. Grid=2^{grid_size}")
    axes[1].set_title(f"KLD Loss. Size={num_obs}. Grid=2^{grid_size}")
    plt.tight_layout()
    plt.show()
    return avg_to_plot


def paper_experiments(sample_experiment,
                      grid_experiment,
                      n_dimensions,
                      n_hidden,
                      means,
                      covars,
                      startprob,
                      transmat,
                      it_experiment=5,
                      lr=0.01,
                      n_epochs=200,
                      random_init=True,
                      log_prob_calc=False,
                      grid_perc_calc=False):
    n_components = n_hidden
    n_features = n_hidden
    sample_size = sample_experiment
    grid_size = 2 ** grid_experiment
    disc_methods = ['random', 'grid', 'LH', 'Sobol', 'Halton']
    loss_functions = ['MSE', 'KLD']
    # Getting the observations for Train and test sets
    mod_opt = Discrete_Model_Optimization(n_components,
                                          n_features,
                                          n_dimensions,
                                          sample_size,
                                          grid_size,
                                          disc_methods[0],
                                          lr,
                                          n_epochs,
                                          startprob,
                                          transmat,
                                          means,
                                          covars,
                                          random_init)
    obs_samples = [mod_opt.observations, mod_opt.observations_test]
    seq_orig = mod_opt.decode(obs_samples[1])[1]
    losses_all = list()
    scores_all = list()
    discr_all = list()
    acc_all = list()
    mods_saved = list()
    if log_prob_calc:
        log_probs_all = list()
    if grid_perc_calc:
        grid_perc_all = list()
    if n_dimensions == 2:
        Q_emps = list()
        Q_ests = list()
        Q_ests_start = list()
    for loss_function in loss_functions:
        for disc_method in disc_methods:
            losses_met = list()
            scores_met = list()
            discr_met = list()
            acc_met = list()
            if log_prob_calc:
                log_probs_met = list()
            if grid_perc_calc:
                grid_perc_met = list()
            message = f'Doing: {disc_method} with {loss_function}. ' \
                f'Sample size={sample_experiment}. Grid size={grid_experiment}'
            print(message)
            for it in range(it_experiment):
                mod_opt = Discrete_Model_Optimization(n_components,
                                                      n_features,
                                                      n_dimensions,
                                                      sample_size,
                                                      grid_size,
                                                      disc_method,
                                                      lr,
                                                      n_epochs,
                                                      startprob,
                                                      transmat,
                                                      means,
                                                      covars,
                                                      random_init,
                                                      obs_samples)
                if (it == it_experiment-1) & (n_dimensions == 2):
                    _, Q_est_start = mod_opt.plot_initial(True)
                    Q_ests_start.append(Q_est_start)
                mods_saved.append(mod_opt)
                params, losses = mod_opt.training(loss_function, verbose=False)
                losses_met.append(losses)
                scores_met.append(mod_opt.score(obs_samples[1]))
                seq_test = mod_opt.decode(obs_samples[1])[1]
                same_state = np.where(seq_orig == seq_test, 1, 0)
                hidd_accuracy = same_state.sum() / len(same_state)
                acc_met.append(hidd_accuracy)
                discr_scale = qmc.scale(mod_opt.discrete_seq,
                                        mod_opt.mins,
                                        mod_opt.maxs,
                                        reverse=True)
                discr = qmc.discrepancy(discr_scale)
                discr_met.append(discr)
                if log_prob_calc:
                    avg_log_prob = eval_estimates(
                        n_components,
                        torch.tensor(mod_opt.observations_test),
                        params[0],
                        params[1])
                    log_probs_met.append(avg_log_prob)
                if grid_perc_calc:
                    grid_perc = len(np.unique(
                        mod_opt.knn_indices)) / mod_opt.grid_size
                    grid_perc_met.append(grid_perc)
            losses_all.append(losses_met)
            scores_all.append(np.mean(scores_met))
            discr_all.append(np.mean(discr_met))
            acc_all.append(np.mean(acc_met))
            if log_prob_calc:
                log_probs_all.append(np.mean(log_probs_met))
            if grid_perc_calc:
                grid_perc_all.append(np.mean(grid_perc_met))
            if n_dimensions == 2:
                Q_emp, Q_est = mod_opt.plot_initial(True)
                Q_emps.append(Q_emp)
                Q_ests.append(Q_est)
    if (n_dimensions == 2) & (it_experiment == 1):
        plot_grids2D(grid_experiment,
                     sample_experiment,
                     Q_emps,
                     Q_ests,
                     Q_ests_start,
                     mods_saved)
    elif (n_dimensions == 3) & (it_experiment == 1):
        plot_grids3D(grid_experiment, sample_experiment, mods_saved)
    plot_metrics(scores_all,
                 acc_all,
                 discr_all,
                 grid_experiment,
                 sample_experiment)
    losses_avg = plot_loss(losses_all, grid_experiment, sample_experiment)
    min_loss = [losses_avg[i][-1] for i in range(len(losses_avg))]
    return min_loss, scores_all, acc_all, discr_all, mods_saved
