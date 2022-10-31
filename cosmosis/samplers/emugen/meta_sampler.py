"""
File that allows us to investigate the properties of a specific run using the
machine-learning accelerated sampling framework
"""

import os
import numpy as np
import json
from matplotlib import pyplot as plt
import seaborn as sns

from .emugen_sampler import EmugenSampler


class MetaSampler:

    def __init__(self, run_name: str, sampler: EmugenSampler, plots_dir: str, true_params: list[float]):
        self.run_name = run_name

        self.sampler = sampler

        # Where to save output plots
        self.plots_dir = plots_dir

        # Strip out ending backslash in plots directory, if it exists
        if self.plots_dir[-1] == '/':
            self.plots_dir = self.plots_dir[:-1]

        # See if the plots directory exists, if not then make it
        if not os.path.isdir(self.plots_dir):
            os.makedirs(self.plots_dir)

        # The values of the true parameters used to generate the original data
        self.true_params = np.asarray(true_params)

    def plot_sample_iterations(self, param_x, param_y):
        """
        Function to plot the sets of samples from the hypercube that is used to train the emulator

        :arg param_x (int) The index of the parameter to plot on the x axis
        :arg param_y (int) The index of the parameter to plot on the y axis
        """
        # Set seabron style
        sns.set_style("darkgrid")

        # Get colormap for number of sample iterations
        cmap = sns.color_palette(palette='inferno', n_colors=self.sampler.iterations, as_cmap=False)

        # First plot with the full hypercube space visible
        fig, ax = plt.subplots(figsize=(11, 9))

        for sample_iter, sample_values in enumerate(self.sampler.full_samples):
            ax.scatter(sample_values[:, param_x], sample_values[:, param_y], alpha=0.6, label=f'Iter {sample_iter + 1}',
                       color=cmap[sample_iter])

        ax.legend()
        fig.tight_layout()
        plt.savefig(f'{self.plots_dir}/samples_scatter_full.pdf')

        # Now plot the same, but with the first set removed
        fig, ax = plt.subplots(figsize=(11, 9))

        for sample_iter, sample_values in enumerate(self.sampler.full_samples[1:], start=1):
            ax.scatter(sample_values[:, param_x], sample_values[:, param_y], alpha=0.6, label=f'Iter {sample_iter + 1}',
                       color=cmap[sample_iter])

        ax.legend()
        fig.tight_layout()
        plt.savefig(f'{self.plots_dir}/samples_scatter_first_removed.pdf')

        # Now set the axes limits such that we zoom in on only the final set of samples
        ax.set_xlim(left=self.sampler.full_samples[3][:, param_x].min(), right=self.sampler.full_samples[3][:, param_x].max())
        ax.set_ylim(bottom=self.sampler.full_samples[3][:, param_y].min(), top=self.sampler.full_samples[3][:, param_y].max())
        fig.tight_layout()
        fig.savefig(f'{self.plots_dir}/samples_scatter_last.pdf')

        plt.close()

    def plot_timings(self, traditional_mcmc_time=None):
        """
        Function to plot the overall runtime of the machine learning accelerated sampler is split between different
        sections
        """
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(0, self.sampler.time_generating, color='cornflowerblue')
        ax.bar(1, self.sampler.time_training, color='slategrey')
        ax.bar(2, self.sampler.time_emcee, color='orange')
        ax.bar(3, self.sampler.time_generating + self.sampler.time_training + self.sampler.time_emcee, color='mediumseagreen')

        labels = ['Sample generation', 'ML training', 'Emcee sampling', 'Total ML']

        if traditional_mcmc_time:
            ax.bar(4, traditional_mcmc_time, color='hotpink')
            labels.append('Traditional MCMC')

        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_ylabel('Wall time (s)')

        ax.set_title(f'Timing breakdown for both ML & traditional MCMC approaches (using {self.sampler.ml_device})')

        fig.tight_layout()
        fig.savefig(f'{self.plots_dir}/timings_bar_plot.pdf')
        plt.close()

    def save_sampler_information(self):
        """
        Function to compute how accurate this machine learning model is
        """

        mean_sq_error = np.sum((self.sampler.chain.mean(0)[:3] - self.true_params) ** 2 / self.true_params)

        self.sampler.properties_dict['mean_sq_error'] = mean_sq_error

        with open(f'./data/{self.run_name}.json', 'w', encoding='utf-8') as data_file:
            json.dump(self.sampler.properties_dict, data_file)

    def plot_loss_history(self):
        sns.set_style("darkgrid")
        cmap = sns.color_palette(palette='inferno', n_colors=self.sampler.iterations, as_cmap=False)

        fig, ax = plt.subplots(figsize=(10, 5))

        for training_iter, taining_loss in enumerate(self.sampler.loss_history):
            ax.semilogy(taining_loss, label=f'Iteration {training_iter}', c=cmap[training_iter])

        ax.set_xlabel('Training iteration')
        ax.set_ylabel('Loss')

        ax.legend()
        fig.tight_layout()

        fig.savefig(f'{self.plots_dir}/loss_history.pdf')
