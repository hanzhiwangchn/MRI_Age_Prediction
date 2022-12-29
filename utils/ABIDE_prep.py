import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os, os.path
import re
import json
import gzip
import shutil
import gc
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    # code for extract and pre-processing ABIDE data

    # unzip original file
    search_path = '../../../Downloads/anat_thickness/'
    results_path = '../../../Downloads/anat_thickness_unzip/'
    file_type = ".gz"
    for fname in os.listdir(path=search_path):
        if fname.endswith(file_type):
            new_fname = fname[:-3]
            with gzip.open(search_path + fname, 'rb') as f_in:
                with open(results_path + new_fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    # select cognitively normal subjects(CN)
    df = pd.read_csv('../../../Downloads/Phenotypic_V1_0b_preprocessed1.csv')
    print(df['DX_GROUP'].value_counts())
    cn_subject_id = df[np.logical_and(df['DX_GROUP'] == 2, df['AGE_AT_SCAN'] < 30)]['SUB_ID'].to_list()
    print(len(cn_subject_id))
    search_path = '../../../Downloads/anat_thickness_unzip/'
    results_path = '../../../Downloads/anat_thickness_cn/'
    for fname in os.listdir(path=search_path):
        subject_id = int(re.findall(r'00[0-9]{5}', fname)[0])
        if subject_id in cn_subject_id:
            shutil.copyfile(search_path+fname, results_path+fname)

    # DIR = './anat_thickness_cn'
    # print(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))

    # transform nii format to numpy
    df = pd.read_csv('../../../Downloads/Phenotypic_V1_0b_preprocessed.csv')
    cn_subject_id = df[df['DX_GROUP'] == 2]['SUB_ID'].to_list()
    cn_subject_age = df[df['DX_GROUP'] == 2]['AGE_AT_SCAN'].to_list()
    assert len(cn_subject_age) == len(cn_subject_id)

    cn_subject_id = [str(i) for i in cn_subject_id]

    dict1 = dict(zip(cn_subject_id, cn_subject_age))
    for k, v in dict1.items():
        dict1[k] = [v]
    search_path = '../../../Downloads/anat_thickness_cn/'
    n = 1
    for fname in os.listdir(path=search_path):
        n += 1
        print(n)
        subject_id = str(int(re.findall(r'00[0-9]{5}', fname)[0]))
        if subject_id in cn_subject_id:
            img = nib.load(search_path + fname)
            img = np.array(img.dataobj)
            print(img.shape)

            img = img[40:-35, 55:-81, 52:-61]
            print(img.shape)

            assert img[0, :, :].sum() == 0
            assert img[-1, :, :].sum() == 0
            assert img[:, 0, :].sum() == 0
            assert img[:, -1, :].sum() == 0
            assert img[:, :, 0].sum() == 0
            assert img[:, :, -1].sum() == 0

            dict1[subject_id].append(np.float32(img))
            cn_subject_id.remove(subject_id)
        gc.collect()

    print(cn_subject_id)
    for i in cn_subject_id:
        del dict1[i]

    age_array = np.array([i[0] for i in dict1.values()])
    print(age_array.shape)
    img_array = np.array([i[1] for i in dict1.values()])
    print(img_array.shape)

    np.save('total_age.npy', age_array)
    np.save('total_images.npy', img_array)

    # load imgaes
    # images = np.load('images.npy')
    # image = images[10]
    # print(image.shape)
    # plt.imshow(image[80, :, :])
    # plt.savefig('1st_dimension_slice.jpg')
    # plt.close()
    # plt.imshow(image[:, 60, :])
    # plt.savefig('2nd_dimension_slice.jpg')
    # plt.close()
    # plt.imshow(image[:, :, 90])
    # plt.savefig('3rd_dimension_slice.jpg')

    # images, df = pickle.load(open('../../camcan/mri_concat.pickle', 'rb'))
    # img = nib.load('../../anat.nii')
    # img = np.array(img.dataobj)
    # print(img.shape)
    # plt.imshow(img[:, :, 100])
    # plt.show()

    # show images
    img = nib.load('../../Downloads/sub')
    a = np.array(img.dataobj)
    # a = a[:100, :, :]
    plt.imshow(a[:, 100, :])
    plt.show()
    print(a.shape)
    print(a.sum())
