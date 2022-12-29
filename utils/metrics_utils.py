import torch
import torch.nn as nn
from scipy import stats


def calculate_correlation_coefficient(preds, labels, args):
    """
    calculate correlation coefficient
    """
    if args.correlation_type == 'pearson':
        error = preds - labels
        vx = error - torch.mean(error)
        vy = labels - torch.mean(labels)
        corr_coef = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    elif args.correlation_type == 'spearman':
        error = preds - labels
        corr_coef = stats.spearmanr(error.cpu(), labels.cpu())[0]
    return corr_coef


def svr(y_pred, y_true, epsilon=1):
    """
    epsilon-insensitive loss. It works in the same way as nn.L1Loss() or nn.MSELoss()
    """
    if y_true.is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    return torch.mean(torch.max(torch.abs(y_pred - y_true) - epsilon,
                                torch.tensor([0. for _ in range(len(y_pred))]).to(device).view(-1, 1)))


class SkewedLossFunction_Ordinary(nn.Module):
    """
    Ordinary skewed loss function implementation.
    """
    def __init__(self, args, lim, median_age):
        super(SkewedLossFunction_Ordinary, self).__init__()
        self.lamda_max = args.init_lambda
        self.lim = lim
        self.median_age = median_age
        self.loss_type = args.loss_type

    def forward(self, y_pred, y_true):
        # we implement skewed loss using an element-wise method, so we need to take the average of it as output
        if self.loss_type == 'L1':
            skewed_loss = SkewedLossFunction_Ordinary.mae_skewed_loss_closure(
                lamda=SkewedLossFunction_Ordinary.linear_adaptive_lamda_closure(
                    lamda_max=self.lamda_max, lim=self.lim, median_age=self.median_age))(y_pred, y_true)
            return torch.mean(skewed_loss)

        elif self.loss_type == 'L2':
            skewed_loss = SkewedLossFunction_Ordinary.mse_skewed_loss_closure(
                lamda=SkewedLossFunction_Ordinary.linear_adaptive_lamda_closure(
                    lamda_max=self.lamda_max, lim=self.lim, median_age=self.median_age))(y_pred, y_true)
            return torch.mean(skewed_loss)

        elif self.loss_type == 'SVR':
            skewed_loss = SkewedLossFunction_Ordinary.svr_skewed_loss_closure(
                lamda=SkewedLossFunction_Ordinary.linear_adaptive_lamda_closure(
                    lamda_max=self.lamda_max, lim=self.lim, median_age=self.median_age))(y_pred, y_true)
            return torch.mean(skewed_loss)

    @staticmethod
    def mae(y_pred, y_true):
        return torch.abs(y_pred - y_true)

    @staticmethod
    def mse(y_pred, y_true):
        return torch.square(y_pred - y_true)

    @staticmethod
    def epsilon_insensitive(y_pred, y_true, epsilon=1):
        # used in skewed loss
        if y_true.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        return torch.max(torch.abs(y_pred - y_true) - epsilon,
                         torch.tensor([0. for _ in range(len(y_pred))]).to(device).view(-1, 1))

    @staticmethod
    def linear_adaptive_lamda_closure(lamda_max, lim, median_age):

        def linear_adaptive_lamda(y):
            """
            The whole age scale is divided into two parts using the median age.
            In each part, returns a λ that linearly depends on the input y.
            λ ranges from -λmax to 0 within the range of lim[0] <= y <= median_age and
            ranges from 0 to +λmax within the range of median_age < y <= lim[1].
            """
            if y.is_cuda:
                device = 'cuda'
            else:
                device = 'cpu'
            corresponding_lambda_values = torch.zeros(size=(len(y), 1)).to(device)
            for each in range(len(y)):
                y_cur = y[each][0]
                if y_cur <= median_age:
                    y_norm_cur = (y_cur - lim[0]) / (median_age - lim[0])
                    corresponding_lambda_values[each][0] = (1 - y_norm_cur) * (-lamda_max)
                else:
                    y_norm_cur = (y_cur - median_age) / (lim[1] - median_age)
                    corresponding_lambda_values[each][0] = y_norm_cur * lamda_max
            return corresponding_lambda_values

        return linear_adaptive_lamda

    @staticmethod
    def mae_skewed_loss_closure(lamda=(lambda x: 0)):
        """
        Function closure
        λ - function that returns a skew parameter. Can be made adaptive to y_true by supplying a function
        """

        def mae_skewed(y_pred, y_true):
            """
            Skewed version of MAE. The skewness is determined by the parameter λ
            """
            return SkewedLossFunction_Ordinary.mae(y_pred, y_true) * \
                torch.exp(torch.sign(y_true - y_pred) * lamda(y_true))
        return mae_skewed

    @staticmethod
    def mse_skewed_loss_closure(lamda=(lambda x: 0)):
        def mse_skewed(y_pred, y_true):
            return SkewedLossFunction_Ordinary.mse(y_pred, y_true) * \
                torch.exp(torch.sign(y_true - y_pred) * lamda(y_true))
        return mse_skewed

    @staticmethod
    def svr_skewed_loss_closure(lamda=(lambda x: 0)):
        def svr_skewed(y_pred, y_true):
            return SkewedLossFunction_Ordinary.epsilon_insensitive(y_pred, y_true) * \
                torch.exp(torch.sign(y_true - y_pred) * lamda(y_true))
        return svr_skewed


class SkewedLossFunction_OneSide(nn.Module):
    """one-side skewed loss function implementation"""
    def __init__(self, args, lim, median_age):
        super(SkewedLossFunction_OneSide, self).__init__()
        self.lamda_max = args.init_lambda
        self.lim = lim
        self.median_age = median_age
        self.loss_type = args.loss_type

    def forward(self, y_pred, y_true):
        if self.loss_type == 'L1':
            skewed_loss = SkewedLossFunction_OneSide.mae_skewed_loss_closure(
                lamda=SkewedLossFunction_OneSide.linear_adaptive_lamda_closure(
                    lamda_max=self.lamda_max, lim=self.lim, median_age=self.median_age))(y_pred, y_true)
            return torch.mean(skewed_loss)

        elif self.loss_type == 'L2':
            skewed_loss = SkewedLossFunction_OneSide.mse_skewed_loss_closure(
                lamda=SkewedLossFunction_OneSide.linear_adaptive_lamda_closure(
                    lamda_max=self.lamda_max, lim=self.lim, median_age=self.median_age))(y_pred, y_true)
            return torch.mean(skewed_loss)

        elif self.loss_type == 'SVR':
            skewed_loss = SkewedLossFunction_OneSide.svr_skewed_loss_closure(
                lamda=SkewedLossFunction_OneSide.linear_adaptive_lamda_closure(
                    lamda_max=self.lamda_max, lim=self.lim, median_age=self.median_age))(y_pred, y_true)
            return torch.mean(skewed_loss)

    @staticmethod
    def mae(y_pred, y_true):
        return torch.abs(y_pred - y_true)

    @staticmethod
    def mse(y_pred, y_true):
        return torch.square(y_pred - y_true)

    @staticmethod
    def epsilon_insensitive(y_pred, y_true, epsilon=1):
        # used in skewed version
        if y_true.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        return torch.max(torch.abs(y_pred - y_true) - epsilon,
                         torch.tensor([0. for _ in range(len(y_pred))]).to(device).view(-1, 1))

    @staticmethod
    def linear_adaptive_lamda_closure(lamda_max, lim, median_age):

        def linear_adaptive_lamda(y):
            if y.is_cuda:
                device = 'cuda'
            else:
                device = 'cpu'
            corresponding_lambda_values = torch.zeros(size=(len(y), 1)).to(device)
            for each in range(len(y)):
                y_cur = y[each][0]
                if y_cur <= median_age:
                    y_norm_cur = (y_cur - lim[0]) / (median_age - lim[0])
                    corresponding_lambda_values[each][0] = (1 - y_norm_cur) * (-lamda_max)
                else:
                    y_norm_cur = (y_cur - median_age) / (lim[1] - median_age)
                    corresponding_lambda_values[each][0] = y_norm_cur * lamda_max
            return corresponding_lambda_values

        return linear_adaptive_lamda

    @staticmethod
    def mae_skewed_loss_closure(lamda=(lambda x: 0)):

        def mae_skewed(y_pred, y_true):
            if y_true.is_cuda:
                device = 'cuda'
            else:
                device = 'cpu'
            return SkewedLossFunction_OneSide.mae(y_pred, y_true) * \
                torch.exp(torch.max(torch.sign(y_true - y_pred) * torch.sign(lamda(y_true)),
                                    torch.tensor([0. for _ in range(len(y_pred))]).to(device).view(-1, 1)) *
                          torch.abs(lamda(y_true)))
        return mae_skewed

    @staticmethod
    def mse_skewed_loss_closure(lamda=(lambda x: 0)):
        def mse_skewed(y_pred, y_true):
            if y_true.is_cuda:
                device = 'cuda'
            else:
                device = 'cpu'
            return SkewedLossFunction_OneSide.mse(y_pred, y_true) * \
                torch.exp(torch.max(torch.sign(y_true - y_pred) * torch.sign(lamda(y_true)),
                                    torch.tensor([0. for _ in range(len(y_pred))]).to(device).view(-1, 1)) *
                          torch.abs(lamda(y_true)))
        return mse_skewed

    @staticmethod
    def svr_skewed_loss_closure(lamda=(lambda x: 0)):
        def svr_skewed(y_pred, y_true):
            if y_true.is_cuda:
                device = 'cuda'
            else:
                device = 'cpu'
            return SkewedLossFunction_OneSide.epsilon_insensitive(y_pred, y_true) * \
                torch.exp(torch.max(torch.sign(y_true - y_pred) * torch.sign(lamda(y_true)),
                                    torch.tensor([0. for _ in range(len(y_pred))]).to(device).view(-1, 1)) *
                          torch.abs(lamda(y_true)))
        return svr_skewed
