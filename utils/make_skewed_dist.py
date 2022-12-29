import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_dataset_camcan():
    """
    load Cam-Can MRI data
    """
    # load MRI data
    images, df = pickle.load(open('../../mri_concat.pickle', 'rb'))
    # reformat data-frame
    df = df.reset_index()

    df_youth = df[df['Age'] <= 40]
    df_mid = df[np.logical_and(df['Age'] > 40, df['Age'] <= 60)]
    df_other = df[df['Age'] > 60]

    df_youth_resample = df_youth.sample(frac=0.3)
    df_mid_resample = df_mid.sample(frac=0.5)
    df_new = pd.concat([df_youth_resample, df_mid_resample, df_other])
    print(df_new.index)

    plt.hist(df_new['Age'])
    plt.show()

    # assign a categorical label to Age for Stratified Split
    df['Age_categorical'] = pd.qcut(df['Age'], 25, labels=[i for i in range(25)])

    new_images = images[df_new.index]
    print(len(new_images))

    with open('camcan_skewed.pickle', 'wb') as f:
        pickle.dump([new_images, df_new], f)

    test_images, test_df = pickle.load(open('camcan_skewed.pickle', 'rb'))
    assert new_images[0][45][45][30] == test_images[0][45][45][30]


def build_dataset_camcan_distribution_shift():
    """
    for distribution shift revision
    """
    # load MRI data
    images, df = pickle.load(open('../../../mri_concat.pickle', 'rb'))
    # reformat data-frame
    df = df.reset_index()

    df_youth = df[df['Age'] <= 40]
    df_mid = df[np.logical_and(df['Age'] > 40, df['Age'] <= 60)]
    df_other = df[df['Age'] > 60]

    df_new = pd.concat([df_mid, df_other])
    df_old = df_youth

    figsize = (20, 7)
    xticks_fontsize = 30
    yticks_fontsize = 30
    xlabel_fontsize = 40
    ylabel_fontsize = 40
    title_fontsize = 45
    plt.figure(figsize=figsize)
    plt.hist(df_new['Age'])
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlabel('Chronological Age (year)', fontsize=xlabel_fontsize)
    plt.ylabel('Number of subjects', fontsize=ylabel_fontsize)
    plt.title('Group 2 & Group 3', fontsize=title_fontsize)
    plt.savefig('1_train.png', bbox_inches='tight')

    plt.figure(figsize=figsize)
    plt.hist(df_old['Age'])
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlabel('Chronological Age (year)', fontsize=xlabel_fontsize)
    plt.ylabel('Number of subjects', fontsize=ylabel_fontsize)
    plt.title('Group 1', fontsize=title_fontsize)
    plt.savefig('1_test.png', bbox_inches='tight')

    df_new = pd.concat([df_other, df_youth])
    # print(df_new.index)
    df_old = df_mid

    plt.figure(figsize=figsize)
    plt.hist(df_new['Age'])
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlabel('Chronological Age (year)', fontsize=xlabel_fontsize)
    plt.ylabel('Number of subjects', fontsize=ylabel_fontsize)
    plt.title('Group 3 & Group 1', fontsize=title_fontsize)
    plt.savefig('2_train.png', bbox_inches='tight')

    plt.figure(figsize=figsize)
    plt.hist(df_old['Age'])
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlabel('Chronological Age (year)', fontsize=xlabel_fontsize)
    plt.ylabel('Number of subjects', fontsize=ylabel_fontsize)
    plt.title('Group 2', fontsize=title_fontsize)
    plt.savefig('2_test.png', bbox_inches='tight')

    df_new = pd.concat([df_mid, df_youth])
    # print(df_new.index)
    df_old = df_other

    plt.figure(figsize=figsize)
    plt.hist(df_new['Age'])
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlabel('Chronological Age (year)', fontsize=xlabel_fontsize)
    plt.ylabel('Number of subjects', fontsize=ylabel_fontsize)
    plt.title('Group 1 & Group 2', fontsize=title_fontsize)
    plt.savefig('3_train.png', bbox_inches='tight')

    plt.figure(figsize=figsize)
    plt.hist(df_old['Age'])
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlabel('Chronological Age (year)', fontsize=xlabel_fontsize)
    plt.ylabel('Number of subjects', fontsize=ylabel_fontsize)
    plt.title('Group 3', fontsize=title_fontsize)
    plt.savefig('3_test.png', bbox_inches='tight')

    # assign a categorical label to Age for Stratified Split
    # df['Age_categorical'] = pd.qcut(df['Age'], 25, labels=[i for i in range(25)])

    # new_images = images[df_new.index]
    # print(len(new_images))
    # old_image = images[df_old.index]

    # with open('camcan_3_train.pickle', 'wb') as f:
    #     pickle.dump([new_images, df_new], f)
    # with open('camcan_3_test.pickle', 'wb') as f:
    #     pickle.dump([old_image, df_old], f)
    #
    # test_images, test_df = pickle.load(open('camcan_3_train.pickle', 'rb'))
    # assert new_images[0][45][45][30] == test_images[0][45][45][30]


if __name__ == '__main__':
    build_dataset_camcan()
