import torch


class DiceLoss:

    def __init__(self):
        pass

    def __call__(self, input, target):
        input = input.reshape(input.size()[0], -1)
        target = target.reshape(target.size()[0], -1).float()
        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1)
        c = torch.sum(target * target, 1)
        d = (2 * a) / (b + c+1e-6)
        return 1 - d


class FocalLoss:

    def __init__(self, gamma=2, alpha=0.25, epsilon=1e-19):
        """

        :param gamma: gamma > 0 Reduces the loss of easily classified samples. Making it easier to focus on difficult, misclassified samples. Larger is more focused on the learning of difficult samples
        :param alpha:Adjust the proportion of positive and negative samples
        :param r : Numerical stability coefficient.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, p, target):
        p_min = p.min()
        p_max = p.max()
        if p_min < 0 or p_max > 1:
            raise ValueError('The range of predicted values should be [0, 1]')
        p = p.reshape(-1, 1)
        target = target.reshape(-1, 1)
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p + self.epsilon)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p + self.epsilon))
        return loss.mean()