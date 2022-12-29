import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import itertools
from performance_tests.testset_performance import print_stats, mae_mean_std, corr_mean_std

os.makedirs('../Plots/ensemble', exist_ok=True)

# ensemble model test using same train/test split

loss_type = ['L1_normal', 'L1_normal_corrected', 'L1_skewed']
model_config = ['resnet']
num_runs = 5
dataset = 'camcan'
result_dir = '../Results/camcan/ResNet'
save_plot_dir = '../Plots/ensemble'


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
                        os.path.join(result_dir, f'{model}_loss_L1_skewed_False_correlation_pearson_'
                                                 f'dataset_{dataset}_run{i}_rnd_state_{random_state}_'
                                                 f'performance_summary.csv'))
                elif each_loss == 'L1_skewed':
                    df_temp = pd.read_csv(
                        os.path.join(result_dir, f'{model}_loss_L1_skewed_True_correlation_pearson_'
                                                 f'dataset_{dataset}_run{i}_rnd_state_{random_state}_'
                                                 f'performance_summary.csv'))
                elif each_loss == 'L1_normal_corrected':
                    df_temp = pd.read_csv(
                        os.path.join(result_dir, f'{model}_loss_L1_skewed_False_correlation_pearson_'
                                                 f'dataset_{dataset}_run{i}_rnd_state_{random_state}_'
                                                 f'corrected_performance_summary.csv'))

                df_temp = df_temp.set_index('ground_truth')
                dfs_temp_2.append(df_temp)
            dfs_temp.append(dfs_temp_2)
        dfs.append(dfs_temp)
    return dfs


def make_plot(dfs, model_config_index, mae_list, corr_list):
    for i in range(len(loss_type)):
        for j in range(len(dfs[i])):
            df0 = dfs[i][j][0].copy(deep=True)
            for k in range(1, num_runs):
                df0[f'predicted_value_{k}'] = dfs[i][j][k]['predicted_value']

            df0['avg_val'] = df0[['predicted_value'] + [f'predicted_value_{m}' for m in range(1, num_runs)]].\
                mean(axis=1)
            df0['avg_std'] = df0[['predicted_value'] + [f'predicted_value_{n}' for n in range(1, num_runs)]].\
                std(axis=1)
            df0['diff'] = df0['avg_val'] - df0.index
            df0['diff_abs'] = df0['diff'].abs()

            mae_list[i].append(df0['diff_abs'].mean())
            corr_list[i].append(np.corrcoef(df0.index, df0['diff'])[0][1])

            plot_ensemble_performance(df0, model_config_index, i, j)
            plot_predicted_error(df0, model_config_index, i, j)
    return mae_list, corr_list


def plot_ensemble_performance(df0, model_config_index, loss_type_index, random_state):
    plt.figure(figsize=(10, 10))
    plt.title(f'{model_config[model_config_index]}_{loss_type[loss_type_index]}_'
              f'{random_state}_performance', fontsize=20)
    plt.scatter(df0.index, df0.index, label='ground truth')
    plt.scatter(df0.index, df0['avg_val'], label=f'predicted value({loss_type[loss_type_index]})')
    plt.annotate(f"MAE:{df0['diff_abs'].mean():.2f}", xy=(30, 50), fontsize=15)
    plt.annotate(f"ADC:{np.corrcoef(df0.index, df0['diff'])[0][1]:.2f}", xy=(30, 40), fontsize=15)
    plt.xlabel('Chronological Age', fontsize=20)
    plt.ylabel('Predicted Age', fontsize=20)
    plt.legend(prop={'size': 10})
    plt.savefig(os.path.join(save_plot_dir,
                             f'{model_config[model_config_index]}_'
                             f'{loss_type[loss_type_index]}_{random_state}_ensemble_performance.jpg'),
                bbox_inches='tight')
    plt.close()


def plot_predicted_error(df0, model_config_index, loss_type_index, random_state):
    plt.figure(figsize=(10, 10))
    plt.title(f'{model_config[model_config_index]}_{loss_type[loss_type_index]}'
              f'{random_state}_predicted_error', fontsize=20)
    plt.hlines(y=0, xmin=10, xmax=90, label='standard', colors='y')
    plt.scatter(df0.index, df0['diff'], label='predicted error')
    plt.xlabel('Chronological Age', fontsize=20)
    plt.ylabel('Predicted error', fontsize=20)
    plt.legend(prop={'size': 10})
    plt.savefig(os.path.join(save_plot_dir,
                             f'{model_config[model_config_index]}_'
                             f'{loss_type[loss_type_index]}_{random_state}_'
                             f'predicted_error.jpg'),
                bbox_inches='tight')
    plt.close()


def analysis_stats_ensemble(metric, analysis_metric, stats_list, desired_pairs):
    # stats_list here is a 3 dimensional array:loss_type, random_state, runs
    stats_list = np.array(stats_list)
    print()
    print(f'{metric} {analysis_metric} test')
    for each_pair in desired_pairs:
        print()
        print(f'{loss_type[each_pair[0]]} & {loss_type[each_pair[1]]}')
        if analysis_metric == 'wilcoxon':
            print(stats.wilcoxon(stats_list[each_pair[0]], stats_list[each_pair[1]]))
        elif analysis_metric == 'ttest':
            print(stats.ttest_rel(stats_list[each_pair[0]], stats_list[each_pair[1]]))


def main(model_config_index):
    mae_list = [[] for _ in range(len(loss_type))]
    corr_list = [[] for _ in range(len(loss_type))]

    dfs = get_model_results(model_config_index)
    # all DataFrames should have the same "ground_truth" value at the same position
    assert dfs[0][0][2].index.values.tolist() == dfs[0][0][4].index.values.tolist()
    assert dfs[1][7][0].index.values.tolist() == dfs[1][7][3].index.values.tolist()
    assert len(dfs) == len(loss_type)
    assert len(dfs[-1]) == 20

    mae_list, corr_list = make_plot(dfs, model_config_index, mae_list, corr_list)
    print_stats(mae_list, category='mae')
    print_stats(corr_list, category='correlation')
    mae_mean_std(mae_list)
    corr_mean_std(corr_list)

    desired_pairs = [(0, 1, 2)]
    desired_pairs = [list(itertools.combinations(i, 2)) for i in desired_pairs]
    desired_pairs = [j for i in desired_pairs for j in i]

    analysis_stats_ensemble(metric='MAE', analysis_metric='wilcoxon',
                            stats_list=mae_list, desired_pairs=desired_pairs)
    analysis_stats_ensemble(metric='MAE', analysis_metric='ttest',
                            stats_list=mae_list, desired_pairs=desired_pairs)
    analysis_stats_ensemble(metric='correlation', analysis_metric='wilcoxon',
                            stats_list=corr_list, desired_pairs=desired_pairs)
    analysis_stats_ensemble(metric='correlation', analysis_metric='ttest',
                            stats_list=corr_list, desired_pairs=desired_pairs)


if __name__ == '__main__':
    model_config_index = 0
    main(model_config_index=model_config_index)
