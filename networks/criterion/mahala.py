import torch

def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)

    return diff, mu

def cov_v(diff, num):
    var = torch.matmul(diff.t(), diff) / num
    return var

def mahalanobis(u, v, cov_x, use_precision=False, reduction=True):
    num, dim = v.size()
    if use_precision == True:
        inv_cov = cov_x
    else:
        inv_cov = torch.inverse(cov_x)
    delta = torch.sub(u, v)
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())

    if reduction:
        return torch.sum(m_loss)/num
    else:
        return m_loss, num
    

def loss_function_mahala(recon_x, x, block_size, cov=None, is_source_list=None, is_target_list=None, update_cov=False, use_precision=False, reduction=True):
    """
    Computes a modified Mahalanobis loss function for anomaly detection tasks.
    This function calculates the loss based on the Mahalanobis distance between 
    reconstructed data (`recon_x`) and the original data (`x`). It also provides 
    an option to update covariance matrices for source and target data, which 
    can be useful for domain adaptation or transfer learning scenarios.
    Parameters:
        recon_x (torch.Tensor): The reconstructed input tensor.
        x (torch.Tensor): The original input tensor.
        block_size (int): The size of the blocks used for reshaping the input tensors.
        cov (torch.Tensor, optional): The covariance matrix or its inverse (precision matrix) 
            used for Mahalanobis distance calculation. Default is None.
        is_source_list (list, optional): A boolean list indicating which samples belong 
            to the source domain. Required if `update_cov` is True. Default is None.
        is_target_list (list, optional): A boolean list indicating which samples belong 
            to the target domain. Required if `update_cov` is True. Default is None.
        update_cov (bool, optional): If True, updates the covariance matrices for source 
            and target data instead of calculating the Mahalanobis loss. Default is False.
        use_precision (bool, optional): If True, assumes `cov` is a precision matrix 
            (inverse of covariance). Default is False.
        reduction (bool, optional): If True, reduces the loss by averaging over dimensions. 
            Default is True.
    Returns:
        torch.Tensor or tuple: 
            - If `update_cov` is False, returns the Mahalanobis loss as a tensor.
            - If `update_cov` is True, returns a tuple containing:
                - The squared difference loss tensor.
                - The updated covariance matrix for the source domain (`cov_diff_source`).
                - The updated covariance matrix for the target domain (`cov_diff_target`), 
                  or None if no target samples are provided.
    Notes:
        - This function modifies the standard Mahalanobis loss by incorporating an 
          option to update covariance matrices for source and target data. This 
          modification is particularly useful in scenarios where the covariance 
          structure of the data changes dynamically, such as in domain adaptation 
          or when working with non-stationary data distributions.
        - The `reduction` parameter allows flexibility in how the loss is aggregated, 
          enabling both element-wise and mean-reduced loss calculations.
    """
    ### Modified mahalanobis loss###
    if update_cov == False:
        loss = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), cov, use_precision, reduction=reduction)
        return loss
    else:
        diff = x - recon_x
        cov_diff_source, _ = cov_v_diff(in_v=diff[is_source_list].view(-1, block_size))

        cov_diff_target = None
        is_calc_cov_target = any(is_target_list)
        if is_calc_cov_target:
            cov_diff_target, _ = cov_v_diff(in_v=diff[is_target_list].view(-1, block_size))

        loss = diff**2
        if reduction:
            loss = torch.mean(loss, dim=1)
        
        return loss, cov_diff_source, cov_diff_target

def loss_reduction_mahala(loss):
    return torch.mean(loss)

def calc_inv_cov(model, device="cpu"):
    inv_cov_source=None
    inv_cov_target=None
    
    cov_x_source = model.cov_source.data
    cov_x_source = cov_x_source.to(device).float()
    inv_cov_source = torch.inverse(cov_x_source)
    inv_cov_source = inv_cov_source.to(device).float()
    
    cov_x_target = model.cov_target.data
    cov_x_target = cov_x_target.to(device).float()
    inv_cov_target = torch.inverse(cov_x_target)
    inv_cov_target = inv_cov_target.to(device).float()
    
    return inv_cov_source, inv_cov_target
