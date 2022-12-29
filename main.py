import argparse
import logging
import os
import torch

from utils.common_utils import RunManager
from utils.model_functions import update_args, build_dataset, build_data_loader, \
    build_model, build_loss_function, train, validation, test, calculate_correlation, \
    update_lamda_max, evaluate_testset_performance, apply_two_stage_correction

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results'
os.makedirs(results_folder, exist_ok=True)

"""
Reference:
H. Wang, M. S. Treder, D. Marshall, D. K. Jones and Y. Li, 
"A Skewed Loss Function for Correcting Predictive Bias in Brain Age Prediction," 
in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3231730.
"""


def build_parser():
    """
    build parser for MRI Age Prediction.
    A template for running the code through the terminal is listed below:
    For the skewed loss, python main.py --skewed-loss --compact-dynamic --comment run0
    For two-stage correction, python main.py --two-stage-correction --comment run1
    """
    parser = argparse.ArgumentParser(description='Brain MRI Age Prediction')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet', 'vgg', 'inception',
                                 'resnet_stride',  'vgg_stride', 'inception_stride'],
                        help='model configurations')
    parser.add_argument('--loss-type', type=str, default='L1', choices=['L1', 'L2', 'SVR'],
                        help='normal loss function configurations')
    parser.add_argument('--correlation-type', type=str, default='pearson', choices=['pearson', 'spearman'],
                        help='correlation metric configurations')
    parser.add_argument('--skewed-loss', action='store_true', default=False,
                        help='use skewed loss function')
    # dynamic lambda strategy config
    parser.add_argument('--compact-dynamic', action='store_true', default=False,
                        help='a compact dynamic-lambda algorithm for the skewed loss')
    parser.add_argument('--compact-target', type=str, default='validation', choices=['train', 'validation'],
                        help='compact dynamic-lambda config: '
                             'specify on which data-set we want the correlation to move toward zero')
    parser.add_argument('--compact-update-interval', type=int, default=10,
                        help='compact dynamic-lambda config: '
                             'update lambda value every a certain number of epoch')
    parser.add_argument('--compact-init-multiplier', type=float, default=1.4,
                        help='compact dynamic-lambda config: '
                             'initialize a multiplier in the stage-2 when updating lambda')
    # apply the two-stage bias correction algorithm
    parser.add_argument('--two-stage-correction', action='store_true', default=False,
                        help='use the two-stage correction approach for the normal loss')
    # frequently used settings
    # NOTE: In the manuscript, we add experiments for distribution shifts.
    #  The updated code is not included in this script for the sake of simplicity.
    parser.add_argument('--dataset', type=str, default='camcan',
                        choices=['camcan', 'camcan_skewed', 'abide', 'abide_symmetric'],
                        help='specify which data-set to use')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--random-state', type=int, default=1000,
                        help='used in train test data-set split')
    parser.add_argument('--comment', type=str, default='run0',
                        help='comments to distinguish different runs')
    # default settings
    parser.add_argument('--val-test-size', type=float, default=0.2,
                        help='proportion of validation & test set of the total data-set')
    parser.add_argument('--test-size', type=float, default=0.5,
                        help='proportion of test set of the "validation & test" set')
    parser.add_argument('--init-lambda', type=float, default=1.0,
                        help='default lambda value for the skewed loss')
    parser.add_argument('--data-aug', action='store_true', default=True,
                        help='Data augmentation especially for MRIs using torchio')
    parser.add_argument('--validation-batch-size', type=int, default=1,
                        help='use 1 as default because of the loss calculation method in RunManager')
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('--params-init', type=str, default='kaiming_uniform',
                        choices=['default', 'kaiming_uniform', 'kaiming_normal'],
                        help='weight initializations')
    parser.add_argument('--acceptable-correlation-threshold', type=float, default=0.05,
                        help='acceptable threshold for correlation when selecting best model')
    parser.add_argument('--save-best', action='store_true', default=True,
                        help='save models with the lowest validation loss in training to prevent over-fitting')
    return parser


def main():
    """overall workflow of MRI Age Prediction"""
    # build parser
    args = build_parser().parse_args()
    # update args based on different data-sets
    args = update_args(args)
    logger.info(f'Parser arguments are {args}')

    # use CUDA if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Found device: {device}')

    # build data-set and data-loader
    dataset_train, dataset_validation, dataset_test, lim, input_shape, median_age = build_dataset(args)
    train_loader, validation_loader, test_loader = \
        build_data_loader(args, dataset_train, dataset_validation, dataset_test)
    logger.info('Dataset loaded')

    # define model and initialize parameters
    net, model_config = build_model(args, device, input_shape)

    # loss function
    loss_fn_train, loss_fn_validation, loss_fn_test = build_loss_function(args, lim, median_age, device)

    # optimizer
    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # build RunManager to save stats from train/validation/test set
    m = RunManager()
    m.begin_run(train_loader, validation_loader, test_loader)

    # initialize best loss(L1) for early stopping on validation set
    best_loss = 8
    # variables for compact dynamic lambda
    lambda_correlation_list = []

    # start training
    logger.info('Start training')
    for epoch in range(1, args.epochs + 1):
        m.begin_epoch()

        train(m, net, device, train_loader, optimizer, loss_fn_train, loss_fn_validation)
        validation(m, net, device, validation_loader, loss_fn_validation)
        test(m, net, device, test_loader, loss_fn_test)
        scheduler.step()

        # calculate correlation on train/validation/test set
        calculate_correlation(args, net, m, train_loader, validation_loader, test_loader, device)

        m.end_epoch()
        m.display_epoch_results()

        assert epoch == m.epoch_num_count

        # dynamic lambda algorithm
        if epoch in range(args.update_lambda_start_epoch, args.epochs+1, args.compact_update_interval) \
                and args.compact_dynamic:
            args, lambda_correlation_list = update_lamda_max(args, m, epoch, lambda_correlation_list)
            # initialize new skewed loss function based on new lamda_max
            loss_fn_train, _, _ = build_loss_function(args, lim, median_age, device)

        # save the model with the best validation loss
        if args.save_best and epoch >= args.save_best_start_epoch:
            if args.skewed_loss:
                # adding correlation threshold as another metric when selecting the best model
                if (m.epoch_stats['validation_loss'] < best_loss) & \
                        (abs(m.epoch_stats['validation_correlation']) <= args.acceptable_correlation_threshold):
                    logger.info(f'Acceptable and lower validation loss found at epoch {m.epoch_num_count}')
                    best_loss = m.epoch_stats['validation_loss']
                    torch.save(net.state_dict(), os.path.join(results_folder, f"{model_config}_Best_Model.pt"))

            else:
                if m.epoch_stats['validation_loss'] < best_loss:
                    logger.info(f'Lower validation loss found at epoch {m.epoch_num_count}')
                    best_loss = m.epoch_stats['validation_loss']
                    torch.save(net.state_dict(), os.path.join(results_folder, f"{model_config}_Best_Model.pt"))

    m.end_run()

    # reinitialize model using the best model parameters and make evaluation on test-set
    evaluate_testset_performance(args, test_loader, device, model_config, results_folder, input_shape)

    # save stats from RunManager
    m.save(os.path.join(results_folder, f'{model_config}_runtime_stats'))

    # apply two-stage correction approach when using normal loss
    if args.two_stage_correction:
        apply_two_stage_correction(args, validation_loader, device, model_config, results_folder, input_shape)

    logger.info('Model finished!')


if __name__ == '__main__':
    main()
