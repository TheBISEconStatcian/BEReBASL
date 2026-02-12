import torch

def accept_based_on_top_percentent_of_arbitrary_var(
        features : torch.Tensor, 
        default_flag : torch.Tensor, 
        var_for_rule : int,
        top_percent : float,
        default_value : int = 1, # 1 or 0
        min_count_bads : int = 4
) -> torch.Tensor:
    if var_for_rule >= features.shape[1]:
        raise ValueError("var_for_rule outside of index")
    
    cutoff = torch.quantile(features[:, var_for_rule], 1 - top_percent)

    accepts = features[:, var_for_rule] >= cutoff


    count_defaults_within_accepts = (default_flag[accepts] == default_value).sum()
    if count_defaults_within_accepts < min_count_bads:
        lidx_defaults_non_accepted = (~accepts) & (default_flag == default_value)
        defaults_still_selectable = lidx_defaults_non_accepted.sum()
        if defaults_still_selectable == 0:
            return accepts
        
        count_bads_to_still_achieve = min_count_bads - count_defaults_within_accepts
        var_for_rule_vals_of_rejected_defaults = features[lidx_defaults_non_accepted][:, var_for_rule]
        
        if defaults_still_selectable <= count_bads_to_still_achieve:
            var_for_rule_vals_of_rejected_defaults = features[lidx_defaults_non_accepted][:, var_for_rule]
            accept_rule_to_include_all_defaults = features[:, var_for_rule] >=  var_for_rule_vals_of_rejected_defaults.min()
            return accept_rule_to_include_all_defaults
        
        new_cutoff = torch.topk(var_for_rule_vals_of_rejected_defaults,k=count_bads_to_still_achieve, largest=True).values[-1]

        return features[:, var_for_rule] >= new_cutoff
    
    return accepts