import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.metrics import roc_auc_score
from loguru import logger

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=11):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        accs = list()
        confs = list()
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):

            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                accs.append(accuracy_in_bin)
                confs.append(avg_confidence_in_bin)

        return ece, accs, confs



class Entropy(nn.Module):
    """
    Calculates the entropy of the distribution and means over batch dimension
    """
    def __init__(self, softmax=True):
        super(Entropy, self).__init__()
        self.softmax = softmax

    def forward(self, logits):
        if self.softmax:
            logits = F.softmax(logits, dim=1)

        entropy = Categorical(logits=logits).entropy().mean()

        return entropy



class AUROC(nn.Module):
    """
    Calculates the AUROC
    (Area under the Receiving Operator Characteristic (ROC) curve)
    for out-of-distribution (OOD) detection
    """
    def __init__(self, softmax=True, equal_size=True):
        super(AUROC, self).__init__()
        self.softmax = softmax
        self.equal_size = equal_size

    def forward(self, id_logits, ood_logits):
        if self.softmax:
            id_logits = F.softmax(id_logits, dim=1)
            ood_logits = F.softmax(ood_logits, dim=1)

        if self.equal_size:
            min_size = np.min((id_logits.shape[0], ood_logits.shape[0]))
            id_logits = id_logits[:min_size,...]
            ood_logits = ood_logits[:min_size,...]

        id_conf_scores, _ = torch.max(id_logits, dim=1, keepdim=False)
        ood_conf_scores, _ = torch.max(ood_logits, dim=1, keepdim=False)

        id_targets = torch.ones_like(id_conf_scores)
        od_targets = torch.zeros_like(ood_conf_scores)

        y_pred = torch.cat((id_conf_scores, ood_conf_scores), dim=0).cpu().data.numpy()
        y_target = torch.cat((id_targets, od_targets), dim=0).cpu().data.numpy()

        score = roc_auc_score(y_target, y_pred)
        logger.info('AUROC score: {:.4f}'.format(score))

        return score





class function_space_analysis(nn.Module):
    def __init__(self, w_softmax=True):
        super(function_space_analysis, self).__init__()
        self.w_softmax = w_softmax
        self.lossFn = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, logits_1, logits_2):

        if not torch.is_tensor(logits_1):
            logits_1 = torch.tensor(logits_1)
            logits_2 = torch.tensor(logits_2)

        if self.w_softmax:
            logits_1 = F.log_softmax(logits_1, dim=1)
            logits_2 = F.log_softmax(logits_2, dim=1)

        distance = self.lossFn(logits_1,logits_2)

        pred_1 = logits_1.max(1, keepdim=True)[1]
        pred_2 = logits_2.max(1, keepdim=True)[1]

        disagreement = torch.sum(pred_1 != pred_2)/pred_2.shape[0]

        return disagreement, distance


def log_gradient(model):
    import wandb
    for name, param in model.named_parameters():
        if param.requires_grad:
            wandb.log({'grad/{}'.format(name[7:]): torch.norm(param.grad)})