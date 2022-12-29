import logging
import pickle
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms

from utils.model_configuration import ResNet, VGG, Inception, ResNet_stride, VGG_stride, Inception_stride
from utils.common_utils import TrainDataset, ValidationDataset, TestDataset, ToTensor_MRI, medical_augmentation
from utils.metrics_utils import svr, calculate_correlation_coefficient, \
    SkewedLossFunction_Ordinary

logger = logging.getLogger(__name__)
results_folder = 'model_ckpt_results'


def update_args(args):
    """
    update args according to different data-sets
    :param args: pre-defined args
    :return: updated args
    """
    if args.dataset == 'camcan':
        args.data_dir = '../camcan/mri_concat.pickle'
        args.epochs = 400
        args.train_batch_size = 32
        args.update_lambda_start_epoch = 150
        args.update_lambda_second_phase_start_epoch = 250
        args.save_best_start_epoch = 100
    elif args.dataset == 'camcan_skewed':
        args.data_dir = '../camcan/camcan_skewed.pickle'
        args.epochs = 400
        args.train_batch_size = 32
        args.update_lambda_start_epoch = 150
        args.update_lambda_second_phase_start_epoch = 250
        args.save_best_start_epoch = 100
    elif args.dataset == 'abide':
        args.data_dir = '../abide/'
        args.epochs = 300
        args.train_batch_size = 16
        args.update_lambda_start_epoch = 75
        args.update_lambda_second_phase_start_epoch = 175
        args.save_best_start_epoch = 50
    elif args.dataset == 'abide_symmetric':
        args.data_dir = '../abide/'
        args.epochs = 300
        args.train_batch_size = 16
        args.update_lambda_start_epoch = 75
        args.update_lambda_second_phase_start_epoch = 175
        args.save_best_start_epoch = 50
    return args


def build_dataset(args):
    """
    build data-set
    :param args: pre-defined args
    :return: Pytorch Data-sets
    """
    if args.dataset == 'camcan':
        dataset_train, dataset_validation, dataset_test, lim, input_shape, median_age = build_dataset_camcan(args)
    elif args.dataset == 'camcan_skewed':
        dataset_train, dataset_validation, dataset_test, lim, input_shape, median_age = build_dataset_camcan(args)
    elif args.dataset == 'abide_symmetric':
        dataset_train, dataset_validation, dataset_test, lim, input_shape, median_age = build_dataset_abide(args)
    elif args.dataset == 'abide':
        dataset_train, dataset_validation, dataset_test, lim, input_shape, median_age = build_dataset_abide(args)
    return dataset_train, dataset_validation, dataset_test, lim, input_shape, median_age


def build_dataset_camcan(args):
    """
    load Cam-Can MRI data
    https://www.cam-can.org
    """
    # load MRI data
    images, df = pickle.load(open(args.data_dir, 'rb'))
    # reformat data-frame
    df = df.reset_index()
    # retrieve the minimum, maximum and median age for skewed loss
    lim = (df['Age'].min(), df['Age'].max())
    median_age = df['Age'].median()
    # add color channel for images (bs, H, D, W) -> (bs, 1, H, D, W)
    images = np.expand_dims(images, axis=1)

    assert len(images.shape) == 5, images.shape
    assert images.shape[1] == 1
    assert len(images) == len(df)

    # assign a categorical label to Age for Stratified Split
    df['Age_categorical'] = pd.qcut(df['Age'], 25, labels=[i for i in range(25)])

    # Stratified train validation-test Split
    split = StratifiedShuffleSplit(test_size=args.val_test_size, random_state=args.random_state)
    train_index, validation_test_index = next(split.split(df, df['Age_categorical']))
    stratified_validation_test_set = df.loc[validation_test_index]
    assert sorted(train_index.tolist() + validation_test_index.tolist()) == list(range(len(df)))

    # Stratified validation test Split
    split2 = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    validation_index, test_index = next(split2.split(stratified_validation_test_set,
                                                     stratified_validation_test_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(validation_index.tolist() + test_index.tolist()) == \
        list(range(len(stratified_validation_test_set.index)))
    assert sorted(validation_index.tolist() + test_index.tolist()) != \
        sorted(list(stratified_validation_test_set.index))

    # get the correct index of original DataFrame for validation/test set
    validation_index = validation_test_index[validation_index]
    test_index = validation_test_index[test_index]

    # ensure there is no duplicated index
    assert sorted(train_index.tolist() + validation_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # get train/validation/test set
    train_images = images[train_index].astype(np.float32)
    validation_images = images[validation_index].astype(np.float32)
    test_images = images[test_index].astype(np.float32)
    # add dimension for labels: (32,) -> (32, 1)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    validation_labels = np.expand_dims(df.loc[validation_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)

    logger.info(f'Training images shape: {train_images.shape}, validation images shape: {validation_images.shape}, '
                f'testing images shape: {test_images.shape}, training labels shape: {train_labels.shape}, '
                f'validation labels shape: {validation_labels.shape}, testing labels shape: {test_labels.shape}')

    # Pytorch Data-set for train set. Apply data augmentation if needed using "torchio"
    if args.data_aug:
        dataset_train = TrainDataset(images=train_images, labels=train_labels,
                                     transform=transforms.Compose([ToTensor_MRI()]),
                                     medical_transform=medical_augmentation)
    else:
        dataset_train = TrainDataset(images=train_images, labels=train_labels,
                                     transform=transforms.Compose([ToTensor_MRI()]))
    # Pytorch Data-set for validation set
    dataset_validation = ValidationDataset(images=validation_images, labels=validation_labels,
                                           transform=transforms.Compose([ToTensor_MRI()]))
    # Pytorch Data-set for test set
    dataset_test = TestDataset(images=test_images, labels=test_labels,
                               transform=transforms.Compose([ToTensor_MRI()]))

    return dataset_train, dataset_validation, dataset_test, lim, train_images.shape[1:], median_age


def build_dataset_abide(args):
    """
    load ABIDE MRI data
    http://preprocessed-connectomes-project.org/abide/download.html
    """
    # load MRI data(in .npy format)
    if args.dataset == 'abide_symmetric':
        images = np.load(args.data_dir + 'symmetric_images.npy')
        age = np.load(args.data_dir + 'symmetric_age.npy')
    elif args.dataset == 'abide':
        images = np.load(args.data_dir + 'total_images.npy')
        age = np.load(args.data_dir + 'total_age.npy')

    df = pd.DataFrame()
    df['Age'] = age
    # retrieve the minimum, maximum and median age for skewed loss
    lim = (df['Age'].min(), df['Age'].max())
    median_age = df['Age'].median()

    # add color channel dimension (bs, H, D, W) -> (bs, 1, H, D, W)
    images = np.expand_dims(images, axis=1)

    assert len(images.shape) == 5, images.shape
    assert images.shape[1] == 1
    assert len(images) == len(df)

    # assign a categorical label to Age for Stratified Split
    df['Age_categorical'] = pd.qcut(df['Age'], 25, labels=[i for i in range(25)])

    # Stratified train validation-test Split
    split = StratifiedShuffleSplit(test_size=args.val_test_size, random_state=args.random_state)
    train_index, validation_test_index = next(split.split(df, df['Age_categorical']))
    stratified_validation_test_set = df.loc[validation_test_index]
    assert sorted(train_index.tolist() + validation_test_index.tolist()) == list(range(len(df)))

    # Stratified validation test Split
    split2 = StratifiedShuffleSplit(test_size=args.test_size, random_state=args.random_state)
    validation_index, test_index = next(split2.split(stratified_validation_test_set,
                                                     stratified_validation_test_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(validation_index.tolist() + test_index.tolist()) == \
        list(range(len(stratified_validation_test_set.index)))
    assert sorted(validation_index.tolist() + test_index.tolist()) != \
        sorted(list(stratified_validation_test_set.index))

    # get the correct index of the original DataFrame for validation/test set
    validation_index = validation_test_index[validation_index]
    test_index = validation_test_index[test_index]

    # ensure there is no duplicated index in 3 data-sets
    assert sorted(train_index.tolist() + validation_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # get train/validation/test set
    train_images = images[train_index].astype(np.float32)
    validation_images = images[validation_index].astype(np.float32)
    test_images = images[test_index].astype(np.float32)
    # add dimension for labels: (32,) -> (32, 1)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    validation_labels = np.expand_dims(df.loc[validation_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)

    logger.info(f'Training images shape: {train_images.shape}, validation images shape: {validation_images.shape}, '
                f'testing images shape: {test_images.shape}, training labels shape: {train_labels.shape}, '
                f'validation labels shape: {validation_labels.shape}, testing labels shape: {test_labels.shape}')

    # Pytorch Data-set for train set. Apply data augmentation if needed using "torchio"
    if args.data_aug:
        dataset_train = TrainDataset(images=train_images, labels=train_labels,
                                     transform=transforms.Compose([ToTensor_MRI()]),
                                     medical_transform=medical_augmentation)
    else:
        dataset_train = TrainDataset(images=train_images, labels=train_labels,
                                     transform=transforms.Compose([ToTensor_MRI()]))
    # Pytorch Data-set for validation set
    dataset_validation = ValidationDataset(images=validation_images, labels=validation_labels,
                                           transform=transforms.Compose([ToTensor_MRI()]))
    # Pytorch Data-set for test set
    dataset_test = TestDataset(images=test_images, labels=test_labels,
                               transform=transforms.Compose([ToTensor_MRI()]))

    return dataset_train, dataset_validation, dataset_test, lim, train_images.shape[1:], median_age


def build_data_loader(args, dataset_train, dataset_validation, dataset_test):
    """make data loader"""
    # build data-loader configurations
    train_kwargs = {'batch_size': args.train_batch_size, 'shuffle': True}
    validation_kwargs = {'batch_size': args.validation_batch_size, 'shuffle': False}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # initialize loader
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, **validation_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    return train_loader, validation_loader, test_loader


def build_model(args, device, input_shape):
    """
    In general, *_stride represents the models using stride=2.
    We implement *_stride architecture to increase the training speed for ABIDE data-sets
    """
    if args.model == 'resnet_stride':
        net = ResNet_stride(input_size=input_shape).to(device)
    elif args.model == 'resnet':
        net = ResNet(input_size=input_shape).to(device)
    elif args.model == 'inception_stride':
        net = Inception_stride(input_size=input_shape).to(device)
    elif args.model == 'inception':
        net = Inception(input_size=input_shape).to(device)
    elif args.model == 'vgg':
        net = VGG(input_size=input_shape).to(device)
    elif args.model == 'vgg_stride':
        net = VGG_stride(input_size=input_shape).to(device)

    # parameter initialization
    def weights_init(m):
        if isinstance(m, nn.Conv3d):
            if args.params_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight.data)
            elif args.params_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight.data)
    if args.params_init != 'default':
        net.apply(weights_init)

    # model_config for output files
    model_config = f'{args.model}_loss_{args.loss_type}_skewed_{args.skewed_loss}_' \
                   f'correlation_{args.correlation_type}_dataset_{args.dataset}_' \
                   f'{args.comment}_rnd_state_{args.random_state}'
    return net, model_config


def build_loss_function(args, lim, median_age, device):
    """build loss functions"""
    # NOTE: no matter what kind of tricks we use in training,
    #  we should always focus the MAE on the test-set.
    loss_fn_validation = nn.L1Loss()
    loss_fn_test = nn.L1Loss()

    # select training loss function
    if args.skewed_loss:
        logger.info(f'Current lambda is {args.init_lambda}')
        loss_fn_train = SkewedLossFunction_Ordinary(args=args, lim=lim, median_age=median_age).to(device)
    else:
        logger.info('Use normal loss')
        if args.loss_type == 'L1':
            loss_fn_train = nn.L1Loss()
        elif args.loss_type == 'L2':
            loss_fn_train = nn.MSELoss()
        elif args.loss_type == 'SVR':
            # no bracket after svr, pls see its implementation
            loss_fn_train = svr
    return loss_fn_train, loss_fn_validation, loss_fn_test


def train(m, net, device, train_loader, optimizer, loss_fn_train, loss_fn_validation):
    """training part"""
    net.train()
    for images, labels in train_loader:
        # zero out gradients
        optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)
        preds = net(images)

        assert preds.shape == labels.shape
        assert len(preds.shape) == 2

        loss = loss_fn_train(preds, labels)

        # loss must be a tensor of a single value
        loss.backward()
        optimizer.step()

        # calculate loss again using validation loss function. It is used to detect over-fitting
        standard_loss = loss_fn_validation(preds, labels)
        # track train loss
        m.track_train_loss(loss=loss)
        m.track_standard_train_loss(loss=standard_loss)


def validation(m, net, device, validation_loader, loss_fn_validation):
    """validation part"""
    net.eval()
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net(images)

            assert preds.shape == labels.shape
            assert len(preds.shape) == 2

            loss = loss_fn_validation(preds, labels)
            # track validation loss
            m.track_validation_loss(loss=loss)


def test(m, net, device, test_loader, loss_fn_test):
    """test part"""
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net(images)

            assert preds.shape == labels.shape
            assert len(preds.shape) == 2

            loss = loss_fn_test(preds, labels)
            # track test loss
            m.track_test_loss(loss=loss)


def calculate_correlation(args, net, m, train_loader, validation_loader, test_loader, device):
    """calculate correlation after current epoch"""
    train_preds_list = []
    train_labels_list = []
    validation_preds_list = []
    validation_labels_list = []
    test_preds_list = []
    test_labels_list = []

    net.eval()
    # training set
    with torch.no_grad():
        for train_images, train_labels in train_loader:
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            train_preds = net(train_images)

            assert train_preds.shape == train_labels.shape
            assert len(train_preds.shape) == 2

            train_preds_list.append(train_preds)
            train_labels_list.append(train_labels)

        # preds and labels will have shape (*, 1)
        preds = torch.cat(train_preds_list, 0)
        labels = torch.cat(train_labels_list, 0)
        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

        correlation = calculate_correlation_coefficient(preds=preds, labels=labels, args=args)
        # track train correlation
        m.collect_train_correlation(correlation=correlation.item())

    # validation set
    with torch.no_grad():
        for val_images, val_labels in validation_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_preds = net(val_images)

            assert val_preds.shape == val_labels.shape
            assert len(val_preds.shape) == 2

            validation_preds_list.append(val_preds)
            validation_labels_list.append(val_labels)

        # preds and labels will have shape (*, 1)
        preds = torch.cat(validation_preds_list, 0)
        labels = torch.cat(validation_labels_list, 0)
        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

        correlation = calculate_correlation_coefficient(preds=preds, labels=labels, args=args)
        # track validation correlation
        m.collect_validation_correlation(correlation=correlation.item())

    # test set
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            test_preds = net(test_images)

            assert test_preds.shape == test_labels.shape
            assert len(test_preds.shape) == 2

            test_preds_list.append(test_preds)
            test_labels_list.append(test_labels)

        # preds and labels will have shape (*, 1)
        preds = torch.cat(test_preds_list, 0)
        labels = torch.cat(test_labels_list, 0)
        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

        correlation = calculate_correlation_coefficient(preds=preds, labels=labels, args=args)
        # track test correlation
        m.collect_test_correlation(correlation=correlation.item())


def moving_average(a, n=3):
    """moving average function"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def update_lamda_max(args, m, epoch, lambda_correlation_list):
    """update lambda value based on correlations"""
    # A moving average function is applied because correlation has wild oscillation.
    # We further select the median value to represent the trends of the correlation
    if args.compact_target == 'train':
        corr_median = np.median(moving_average(m.run_correlation_train[-1 * args.compact_update_interval:], n=3))
    elif args.compact_target == 'validation':
        corr_median = np.median(moving_average(m.run_correlation_validation[-1 * args.compact_update_interval:], n=3))
    logger.info(f'median averaged correlation is {corr_median}')
    temp_lambda_corr_pair = [args.init_lambda, corr_median]

    lambda_correlation_list.append(temp_lambda_corr_pair)
    # lambda_correlation_list only keeps the last 10 pairs of results for updating lambda
    if len(lambda_correlation_list) > 10:
        lambda_correlation_list.pop(0)

    # At the 1st phase of dynamic lambda, we use a naive approach to tune lambda to get correlations
    # for different lambda values.
    # At the 2nd phase, we apply linear regression to find the optimal lambda value
    if epoch >= args.update_lambda_second_phase_start_epoch:
        args.init_lambda = find_optimal_lambda(lambda_correlation_list)
    else:
        # Instead of using a single multiplier, we assign small changes on it to improve stability
        if corr_median < -0.1:
            args.init_lambda = args.init_lambda * torch.normal(mean=torch.tensor([args.compact_init_multiplier]),
                                                               std=torch.tensor([0.05])).item()
        elif corr_median > 0.1:
            args.init_lambda = args.init_lambda / torch.normal(mean=torch.tensor([args.compact_init_multiplier]),
                                                               std=torch.tensor([0.05])).item()

    logger.info(f'updated lambda at epoch:{epoch} is {args.init_lambda}')
    return args, lambda_correlation_list


def find_optimal_lambda(lambda_correlation_list):
    """
    find the best lambda value to make correlation move towards zero using LR
    """
    lambda_correlation_array = np.array(lambda_correlation_list)
    lambda_val = lambda_correlation_array[:, 0]
    correlation = lambda_correlation_array[:, 1]

    # use linear regression as a start
    lr = LinearRegression()
    lr.fit(lambda_val.reshape(-1, 1), correlation.reshape(-1, 1))

    test_lambda_val = np.array([0, 1])
    test_correlation_pred = lr.predict(test_lambda_val.reshape(-1, 1))
    # get the optimal value
    slope = test_correlation_pred[1] - test_correlation_pred[0]
    bias = test_correlation_pred[0]
    # if slope becomes zero, it means the dots are on a horizontal line,
    # which will result in a much larger lambda.
    if abs(slope[0]) < 1e-2:
        opt_lambda = np.mean(lambda_val)
    else:
        opt_lambda = -1 * bias[0] / slope[0]

    # lambda should not be a negative value
    # stability improvement
    if opt_lambda < 0:
        opt_lambda = 0.0
    if opt_lambda > 20:
        opt_lambda = 20.0

    logger.info(f'slope of lr is {slope}; bias of lr is {bias}')
    logger.info(f'optimal lambda is {opt_lambda}')
    return opt_lambda


def evaluate_testset_performance(args, test_loader, device, model_config, results_folder, input_shape):
    """evaluate performance here, we need to load the best model instead of the model at epoch 300"""
    if args.model == 'resnet_stride':
        net_test = ResNet_stride(input_size=input_shape).to(device)
    elif args.model == 'resnet':
        net_test = ResNet(input_size=input_shape).to(device)
    elif args.model == 'inception_stride':
        net_test = Inception_stride(input_size=input_shape).to(device)
    elif args.model == 'inception':
        net_test= Inception(input_size=input_shape).to(device)
    elif args.model == 'vgg':
        net_test = VGG(input_size=input_shape).to(device)
    elif args.model == 'vgg_stride':
        net_test = VGG_stride(input_size=input_shape).to(device)

    net_test.load_state_dict(state_dict=torch.load(os.path.join(results_folder, f"{model_config}_Best_Model.pt")))
    net_test.eval()
    # save results to csv file
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net_test(images)

            assert preds.shape == labels.shape

            preds_list.append(preds)
            labels_list.append(labels)

        # preds and labels will have shape (*, 1)
        preds_tensor = torch.cat(preds_list, 0)
        labels_tensor = torch.cat(labels_list, 0)

        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

    df_save = pd.DataFrame()
    df_save['predicted_value'] = preds_tensor.squeeze().cpu().numpy()
    df_save['ground_truth'] = labels_tensor.squeeze().cpu().numpy()
    df_save.to_csv(os.path.join(results_folder, f'{model_config}_performance_summary.csv'))


def apply_two_stage_correction(args, validation_loader, device, model_config, results_folder, input_shape):
    """apply two-stage correction on validation set"""
    # initialize best model
    if args.model == 'resnet_stride':
        net_test = ResNet_stride(input_size=input_shape).to(device)
    elif args.model == 'resnet':
        net_test = ResNet(input_size=input_shape).to(device)
    elif args.model == 'inception_stride':
        net_test = Inception_stride(input_size=input_shape).to(device)
    elif args.model == 'inception':
        net_test= Inception(input_size=input_shape).to(device)
    elif args.model == 'vgg':
        net_test = VGG(input_size=input_shape).to(device)
    elif args.model == 'vgg_stride':
        net_test = VGG_stride(input_size=input_shape).to(device)

    # reload weights
    net_test.load_state_dict(state_dict=torch.load(os.path.join(results_folder, f"{model_config}_Best_Model.pt")))
    net_test.eval()

    df_test = pd.read_csv(os.path.join(results_folder, f"{model_config}_performance_summary.csv"))

    # save validation results to DataFrame
    val_preds_list = []
    val_labels_list = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            preds = net_test(images)

            assert preds.shape == labels.shape

            val_preds_list.append(preds)
            val_labels_list.append(labels)

        # preds and labels will have shape (*, 1)
        val_preds_tensor = torch.cat(val_preds_list, 0)
        val_labels_tensor = torch.cat(val_labels_list, 0)

        assert preds.shape == labels.shape
        assert preds.shape[1] == 1

    df_validation = pd.DataFrame()
    df_validation['predicted_value'] = val_preds_tensor.squeeze().cpu().numpy()
    df_validation['ground_truth'] = val_labels_tensor.squeeze().cpu().numpy()

    validation_slope, validation_bias = two_stage_linear_fit(df_val=df_validation)
    two_steps_bias_correction(validation_slope, validation_bias, df_test, model_config)


def two_stage_linear_fit(df_val):
    """two-stage approach: linear fit on validation set"""
    predicted_value = df_val['predicted_value'].values
    ground_truth = df_val['ground_truth'].values

    # use linear regression
    lr = LinearRegression()
    lr.fit(ground_truth.reshape(-1, 1), predicted_value.reshape(-1, 1))

    test_val = np.array([0, 1])
    test_pred = lr.predict(test_val.reshape(-1, 1))

    slope = test_pred[1] - test_pred[0]
    bias = test_pred[0]
    return slope[0], bias[0]


def two_steps_bias_correction(slope, bias, df_test, model_config):
    """two-stage approach: correction on predicted value to get unbiased prediction"""
    df_test['predicted_value'] = df_test['predicted_value'].apply(lambda x: (x-bias) / slope)
    df_test.to_csv(os.path.join(results_folder, f"{model_config}_corrected_performance_summary.csv"))
