import torch
from pycox.models.loss import cox_ph_loss


def cox_loss(log_h: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Custom loss function that uses Cox Proportional Hazards from pycox
    
    Args:
        log_h (tensor): model output representing log risk score for n patients
        durations (tensor): true survival times (OS.time) for n patients
        events (tensor): true survival outcomes (OS), either 1=died or 0=censored, for n patients
        
    Returns:
        loss (tensor): CoxPH loss values for each sample
    """
    
    # ensure we have the right dtypes
    log_h = log_h.float()
    durations = durations.float()
    
    # edge case: batch is all censored (every event is 0)
    # loss is undefined, so add a tiny value to each event so it is near-zero
    if events.sum() == 0:
        events = events + 1e-6
    
    # pycox handles sorting and ties for us
    loss = cox_ph_loss(log_h, durations, events)
    
    return loss