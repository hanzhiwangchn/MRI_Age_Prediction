import os
import numpy as np
import pandas as pd
from scipy import stats
import itertools
from matplotlib import pyplot as plt

# Test-set MAE and correlation evaluation

loss_type = ['L1_normal', 'L1_normal_corrected', 'L1_skewed']
model_config = ['resnet_stride']
num_runs = 5
dataset = 'abide_symmetric'
result_dir = '../Results/abide_symmetric/ResNet'
save_plot_dir = '../Plots/temp'


def get_model_results(model_config_index):
    """read csv files for all models and all runs"""
    dfs = []
    model = model_config[model_config_index]
    for each_loss in loss_type:
        dfs_temp = []
        for random_state in range(1000, 1020):
            dfs_temp_2 = []
            for i in range(num_runs):
                if each_loss == 'L1_normal':

                    df_temp = pd.read_csv(
                        os.path.join(result_dir, f'{model}_loss_L1_skewed_False_'
                                                 f'correlation_pearson_dataset_{dataset}_'
                                                 f'run{i}_rnd_state_{random_state}_'
                                                 f'performance_summary.csv'))
                elif each_loss == 'L1_skewed':
                    df_temp = pd.read_csv(
                        os.path.join(result_dir, f'{model}_loss_L1_skewed_True_'
                                                 f'correlation_pearson_dataset_{dataset}_'
                                                 f'run{i}_rnd_state_{random_state}_'
                                                 f'performance_summary.csv'))
                elif each_loss == 'L1_normal_corrected':
                    df_temp = pd.read_csv(
                        os.path.join(result_dir, f'{model}_loss_L1_skewed_False_'
                                                 f'correlation_pearson_dataset_{dataset}_'
                                                 f'run{i}_rnd_state_{random_state}_'
                                                 f'corrected_performance_summary.csv'))

                df_temp['error'] = df_temp['predicted_value'] - df_temp['ground_truth']
                df_temp['abs_error'] = df_temp['error'].abs()
                dfs_temp_2.append(df_temp)
            dfs_temp.append(dfs_temp_2)
        dfs.append(dfs_temp)
    return dfs


def calculate_correlation(dfs):
    corr_list = []
    for i in range(len(dfs)):
        temp = []
        for j in range(len(dfs[i])):
            temp_inner = []
            for k in range(len(dfs[i][j])):
                # corr = float(f"{stats.spearmanr(dfs[i][j][k]['ground_truth'], dfs[i][j][k]['error'])[0]:.2f}")
                corr = float(f"{np.corrcoef(dfs[i][j][k]['ground_truth'], dfs[i][j][k]['error'])[0][1]:.2f}")
                temp_inner.append(corr)
            temp.append(temp_inner)
        corr_list.append(temp)
    assert len(corr_list) == len(dfs)
    return corr_list


def calculate_mae(dfs):
    mae_list = []
    for i in range(len(dfs)):
        temp = []
        for j in range(len(dfs[i])):
            temp_inner = []
            for k in range(len(dfs[i][j])):
                mae = float(f"{dfs[i][j][k]['abs_error'].mean():.2f}")
                temp_inner.append(mae)
            temp.append(temp_inner)
        mae_list.append(temp)
    assert len(mae_list) == len(dfs)
    return mae_list


def print_stats(stats_list, category='mae'):
    print()
    if category == 'mae':
        print('MAE summary')
    elif category == 'correlation':
        print("Correlation summary")
    for i in range(len(loss_type)):
        print()
        print(loss_type[i])
        print(stats_list[i])


def analysis_stats(metric, analysis_metric, stats_list, desired_pairs, method):
    # stats_list here is a 3 dimensional array:loss_type, random_state, runs
    stats_list = np.array(stats_list)
    print()
    print(f'{metric} {analysis_metric} test')
    if method == 'individual':
        # 100 models test
        stats_list = stats_list.reshape(stats_list.shape[0], -1)
    elif method == 'average':
        stats_list = np.mean(stats_list, axis=2)
        print(stats_list)
    elif method == 'ensemble':
        # ensemble test is implemented in the ensemble_performance_test.py
        print('pls check the method')
        exit(0)
    for each_pair in desired_pairs:
        print()
        print(f'{loss_type[each_pair[0]]} & {loss_type[each_pair[1]]}')
        if analysis_metric == 'wilcoxon':
            print(stats.wilcoxon(stats_list[each_pair[0]], stats_list[each_pair[1]]))
        elif analysis_metric == 'ttest':
            print(stats.ttest_rel(stats_list[each_pair[0]], stats_list[each_pair[1]]))


def mae_mean_std(mae_list):
    print()
    mae_list = np.array(mae_list)
    mae_list = mae_list.reshape(mae_list.shape[0], -1)
    means = mae_list.mean(axis=1)
    stds = mae_list.std(axis=1)
    assert means.shape == (len(loss_type), )
    for i in range(len(loss_type)):
        print(f'{loss_type[i]}: {means[i]}, {stds[i]}')


def corr_mean_std(corr_list):
    print()
    corr_list = np.array(corr_list)
    corr_list = corr_list.reshape(corr_list.shape[0], -1)
    means = corr_list.mean(axis=1)
    stds = corr_list.std(axis=1)
    assert means.shape == (len(loss_type), )
    for i in range(len(loss_type)):
        print(f'{loss_type[i]}: {means[i]}, {stds[i]}')


def get_comparison_plots(mae_list, corr_list):
    plt.figure()
    for each_loss_idx in range(len(loss_type)):
        plt.scatter(list(range(1, 1 + num_runs)), mae_list[each_loss_idx], label=f'{loss_type[each_loss_idx]}')
    plt.xlabel('times', fontsize=20)
    plt.ylabel('MAE', fontsize=20)
    plt.legend(prop={'size': 10})
    plt.savefig(os.path.join(save_plot_dir,
                             f'{"MAE_comparison.jpg"}'),
                bbox_inches='tight')
    plt.close()
    plt.figure()
    for each_loss_idx in range(len(loss_type)):
        plt.scatter(list(range(1, 1 + num_runs)), corr_list[each_loss_idx], label=f'{loss_type[each_loss_idx]}')
    plt.xlabel('times', fontsize=20)
    plt.ylabel('correlation', fontsize=20)
    plt.hlines(y=0, xmin=1, xmax=5, colors='y')
    plt.legend(prop={'size': 10})
    plt.savefig(os.path.join(save_plot_dir,
                             f'{"corr_comparison.jpg"}'),
                bbox_inches='tight')
    plt.close()


def main(model_config_index):
    dfs = get_model_results(model_config_index)

    # assert dfs[0][7]['ground_truth'].values.tolist() == dfs[1][7]['ground_truth'].values.tolist()
    # assert dfs[1][7]['ground_truth'].values.tolist() != dfs[2][4]['ground_truth'].values.tolist()
    # assert len(dfs) == len(loss_type)
    # assert len(dfs[-1]) == num_runs

    corr_list = calculate_correlation(dfs)
    mae_list = calculate_mae(dfs)
    print_stats(mae_list, category='mae')
    print_stats(corr_list, category='correlation')

    desired_pairs = [(0, 1, 2)]
    desired_pairs = [list(itertools.combinations(i, 2)) for i in desired_pairs]
    desired_pairs = [j for i in desired_pairs for j in i]

    method = 'average'
    analysis_stats(metric='MAE', analysis_metric='wilcoxon',
                   stats_list=mae_list, desired_pairs=desired_pairs, method=method)
    analysis_stats(metric='MAE', analysis_metric='ttest',
                   stats_list=mae_list, desired_pairs=desired_pairs, method=method)
    analysis_stats(metric='correlation', analysis_metric='wilcoxon',
                   stats_list=corr_list, desired_pairs=desired_pairs, method=method)
    analysis_stats(metric='correlation', analysis_metric='ttest',
                   stats_list=corr_list, desired_pairs=desired_pairs, method=method)

    mae_mean_std(mae_list)
    corr_mean_std(corr_list)


if __name__ == '__main__':
    model_config_index = 0
    main(model_config_index)
