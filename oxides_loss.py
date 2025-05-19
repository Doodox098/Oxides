import torch
from OxideModel import OxideModel


def oxides_loss(oxide_models, global_shift, reference, it, config):
    oxygen, time = reference
    t_begin_weight = config['t_begin_weight']
    t_begin_change_weight = config['t_begin_change_weight']
    guaranteed_oxides = config['guaranteed_oxides']
    guaranteed_usage_weight = config['guaranteed_usage_weight']
    oxide_usage_weight = config['oxide_usage_weight']
    residual_weight = config['residual_weight']
    residual = oxygen
    begin_loss_all = 0
    usage_loss_all = 0
    begin_change_loss_all = 0
    for oxide_name, oxide in oxide_models.items():
        Value = oxide(time, global_shift)
        # T_beg loss
        TB = oxide.get_t_beg() + global_shift
        begin_sigmoid = 1 / (1 + torch.exp(time - TB))  # Gives smooth "time < TB" function
        if oxide.get_v_max() < 0.2 or it < 200:
            begin_sigmoid = begin_sigmoid.detach()
        begin_loss = t_begin_weight * torch.mean(Value * begin_sigmoid)
        begin_change_loss = t_begin_change_weight * torch.abs(oxide.T_beg_delta)
        # Usage loss
        usage_loss = 0
        if oxide_name not in guaranteed_oxides:
            usage_loss = oxide_usage_weight * oxide.get_v_max()
        else:
            usage_loss = guaranteed_usage_weight * oxide.get_v_max()
        begin_change_loss_all += begin_change_loss
        begin_loss_all += begin_loss
        usage_loss_all += usage_loss
        # Residual loss
        residual = residual - Value
    residual_loss = residual_weight * torch.mean(torch.abs(residual))
    if torch.isnan(residual_loss):
        print("NaN detected in residual loss!")
    return begin_change_loss_all, begin_loss_all, usage_loss_all, residual_loss