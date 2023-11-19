import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('multilabel_cross_entropy')
def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """
    if cfg.dataset.task_type == 'classification_multilabel':
        if cfg.model.loss_fun != 'cross_entropy':
            raise ValueError("Only 'cross_entropy' loss_fun supported with "
                             "'classification_multilabel' task_type.")
        bce_loss = nn.BCEWithLogitsLoss()
        
        # Reshape true to match the shape of pred
        true = true.unsqueeze(0) if true.dim() == 1 else true
        
        is_labeled = true == true  # Filter our nans.
        return bce_loss(pred[is_labeled], true[is_labeled].float()), pred
