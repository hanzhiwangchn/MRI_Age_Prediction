import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# plot correlation changes on train/val/test set in the training process
# We only generate the ResNet performances on the Cam-CAN data-set for illustrations

loss_type = ['L1 loss', 'skewed L1 loss']
num_runs = 1
model_config = ['ResNet']
dataset = 'camcan'
result_dir = '../model_ckpt_results'
save_plot_dir = '../Plots/temp'


def get_model_results(model_config_index, random_state):
    """read csv files for all models and all runs"""
    dfs = []
    model = model_config[model_config_index]
    for each_loss in loss_type:
        dfs_temp = []
        for i in range(num_runs):
            if each_loss == 'L1 loss':
                df_temp = pd.read_csv(
                    os.path.join(result_dir, f'{model}_loss_L1_skewed_False_'
                                             f'correlation_pearson_dataset_{dataset}_'
                                             f'run{i}_rnd_state_{random_state}_runtime_stats.csv'))
            if each_loss == 'skewed L1 loss':
                df_temp = pd.read_csv(
                    os.path.join(result_dir, f'{model}_loss_L1_skewed_True_'
                                             f'correlation_pearson_dataset_{dataset}_'
                                             f'run{i}_rnd_state_{random_state}_runtime_stats.csv'))
            dfs_temp.append(df_temp)
        dfs.append(dfs_temp)
    return dfs


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def make_shaded_line(dfs, loss_type, random_state, model_config, start_epoch):
    # plot correlation changes
    plt.figure(figsize=(25, 9))
    plt.title(f'ADC Comparison', fontsize=40)
    for metric in ['val_correlation']:
        if metric == 'val_correlation':
            for loss_idx in range(len(loss_type)):
                mean = np.mean(np.array([moving_average(dfs[loss_idx][run_idx].
                                                        loc[start_epoch:, 'validation_correlation'].values)
                                         for run_idx in range(5)]), axis=0)
                std = np.std(np.array([moving_average(dfs[loss_idx][run_idx].
                                                      loc[start_epoch:, 'validation_correlation'].values)
                                       for run_idx in range(5)]), axis=0)

                x = np.arange(len(mean))
                if loss_idx == 0:
                    plt.plot(x, mean, label='L1 loss', linestyle='--',
                             marker='o', color='b', markersize=9, linewidth=3)
                elif loss_idx == 1:
                    plt.plot(x, mean, label='Skewed L1 loss', color='r', linewidth=3)
                if loss_idx == 0:
                    plt.fill_between(x, mean - std, mean + std, color='b', alpha=0.20)
                else:
                    plt.fill_between(x, mean - std, mean + std, color='r', alpha=0.20)
                plt.hlines(y=0, xmin=start_epoch, xmax=400)
                plt.legend(prop={'size': 35})
                plt.xlabel('epoch', fontsize=40)
                plt.ylabel('ADC', fontsize=40)
                plt.tick_params(axis='both', which='major', labelsize=30)
            plt.savefig(os.path.join(save_plot_dir,
                                     f'shaded_correlation_comparison_plot_{model_config[model_config_idx]}_'
                                     f'{loss_type[loss_idx]}_{random_state}.jpg'), bbox_inches='tight')
            plt.close()


def make_plot_comparison(dfs, loss_type, random_state, model_config, start_epoch):
    # plot correlation changes between normal loss and skewed loss
    for run_idx in range(num_runs):
        plt.figure(figsize=(25, 9))
        plt.title(f'ADC Comparison', fontsize=40)
        for metric in ['val_correlation']:
            # if metric == 'train_correlation':
            #     plt.subplot(3, 1, 1)
            #     for loss_idx in range(len(loss_type)):
            #         plt.plot(moving_average(dfs[loss_idx][run_idx].loc[start_epoch:, 'train_correlation'].values),
            #                  label=f'{loss_type[loss_idx]}_train')
            if metric == 'val_correlation':
                # plt.subplot(3, 1, 2)
                for loss_idx in range(len(loss_type)):
                    # L1 loss
                    if loss_idx == 0:
                        plt.plot(moving_average(dfs[loss_idx][run_idx].loc[start_epoch:, 'validation_correlation'].values),
                                 label='L1 loss', linestyle='--', marker='o', color='b', markersize=10, linewidth=3)
                    # skewed L1 loss
                    if loss_idx == 1:
                        plt.plot(moving_average(dfs[loss_idx][run_idx].loc[start_epoch:, 'validation_correlation'].values),
                                 label='Skewed L1 loss', color='r', linewidth=3)
            # if metric == 'test_correlation':
            #     # plt.subplot(3, 1, 3)
            #     for loss_idx in range(len(loss_type)):
            #         plt.plot(moving_average(dfs[loss_idx][run_idx].loc[start_epoch:, 'test_correlation'].values),
            #                  label=f'{loss_type[loss_idx]}')

            plt.hlines(y=0, xmin=start_epoch, xmax=400)
            plt.legend(prop={'size': 35})
            plt.xlabel('epoch', fontsize=40)
            plt.ylabel('ADC', fontsize=40)
            plt.tick_params(axis='both', which='major', labelsize=30)
        plt.savefig(os.path.join(save_plot_dir,
                                 f'correlation_comparison_plot_{model_config[model_config_idx]}_'
                                 f'{loss_type[loss_idx]}_{random_state}_'
                                 f'{run_idx}.jpg'), bbox_inches='tight')
        plt.close()


def make_plot_single_run(dfs, loss_type, random_state, model_config, start_epoch):
    for loss_idx in range(len(loss_type)):
        for run_idx in range(num_runs):
            plt.figure(figsize=(25, 25))
            plt.title(f'ADC plot_{random_state}', fontsize=40)
            for metric in ['train_correlation', 'val_correlation', 'test_correlation']:
                # if metric == 'train_correlation':
                #     plt.subplot(3, 1, 1)
                #     plt.plot(moving_average(dfs[loss_idx][run_idx].loc[start_epoch:, 'train_correlation'].values),
                #              label=f'{loss_type[loss_idx]}_train')
                if metric == 'val_correlation':
                    # plt.subplot(3, 1, 2)
                    plt.plot(moving_average(dfs[loss_idx][run_idx].loc[start_epoch:, 'validation_correlation'].values),
                             label=f'{loss_type[loss_idx]}_validation')
                # if metric == 'test_correlation':
                #     plt.subplot(3, 1, 3)
                #     plt.plot(moving_average(dfs[loss_idx][run_idx].loc[start_epoch:, 'test_correlation'].values),
                #              label=f'{loss_type[loss_idx]}_test')

                plt.hlines(y=0, xmin=start_epoch, xmax=400)
                plt.legend(prop={'size': 10})
                plt.xlabel('epoch', fontsize=15)
                plt.ylabel('ADC', fontsize=15)
                plt.tick_params(axis='both', which='major', labelsize=15)
            plt.savefig(os.path.join(save_plot_dir,
                                     f'correlation_plot_{model_config[model_config_idx]}_'
                                     f'{loss_type[loss_idx]}_{random_state}_'
                                     f'{run_idx}.jpg'), bbox_inches='tight')
            plt.close()


def main(model_config_idx):
    for random_state in range(1001, 1002):
        dfs = get_model_results(model_config_idx, random_state)

        assert len(dfs) == len(loss_type)
        assert len(dfs[-1]) == num_runs

        # comparisons between the normal loss and the skewed loss
        make_plot_comparison(dfs, loss_type, random_state, model_config, start_epoch=0)

        # comparisons between the normal loss and the skewed loss (shaded lines)
        # make_shaded_line(dfs, loss_type, random_state, model_config, start_epoch=0)

        # comparisons among training, validation and test set
        # make_plot_single_run(dfs, loss_type, random_state, model_config, start_epoch=0)


if __name__ == '__main__':
    model_config_idx = 0
    main(model_config_idx)
