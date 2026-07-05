from collections import defaultdict

all_threshold_lbls = [th + '_' + m for th in THRESHOLD_BASIS for m in METRIC_CATEGORIES]
for path, current_simulation_objects in all_sim_objs.items():
    print("Starting", path)
    dgp = current_simulation_objects["data_generator"]
    credit_data: CreditData = deepcopy(current_simulation_objects['credit_data'])
    (
        feats_all,     # [..., N, F]
        lbls_all,      # [..., N]
        gen_round_all, # [..., N]
        acc_flag_all   # [..., N]
    ) = credit_data.unbiased_obs(include_gen_round=True, include_accepted_status=True)

    acc_flags_per_th_all = {k: v.clone() for k, v in current_simulation_objects["alternative_accepted"].items()}
    lbl_acc_flag_in_data = set(all_threshold_lbls) - set(acc_flags_per_th_all.keys())
    assert len(lbl_acc_flag_in_data) == 1
    lbl_acc_flag_in_data = next(lbl_acc_flag_in_data.__iter__())
    acc_flags_per_th_all[lbl_acc_flag_in_data] = acc_flag_all

    arange_gen_round = torch.arange(credit_data.last_gen_round+1, device=credit_data.device) # [G]
    G = arange_gen_round.size(0)
    gen_round_broadcast = gen_round_all.unsqueeze(-2) # [..., 1, N]
    arange_gen_round_broadcast = arange_gen_round.unsqueeze(-1) # [G, 1]
    mask_valid_round_cummulative_all = gen_round_broadcast <= arange_gen_round_broadcast # [..., G, N]
    mask_valid_round_window_all = (
        (arange_gen_round_broadcast <= gen_round_broadcast) &
        (gen_round_broadcast < (arange_gen_round_broadcast + 5))
    ) # [..., G, N]
    gather_idxs_per_round_cummulative_all = mask_valid_round_cummulative_all.cumsum(dim=-1)-1 # [..., G, N]

    N = lbls_all.size(-1)

    total_counts_per_round = credit_data.counts_per_round_as_dict()["total"]
    

    K, CV = current_simulation_objects["k_folds"], current_simulation_objects["cv_count"]

    stats_perf_bayes_all = defaultdict(list)

    for i, (
        gen_round,
        gather_idxs_per_round_cummulative,
        mask_valid_round_cummulative,
        mask_valid_window_simple,
        feats,
        lbls,
        acc_flag,
        counts_per_round
    ) in enumerate(zip(
        *[
            ten.view(-1, G, N)
            if ten.size(-2) == G else
            ten.view(-1, N, *([] if ten.shape != feats_all.shape else feats_all.shape[-1:]))
            for ten in (
                gen_round_all,
                gather_idxs_per_round_cummulative_all,
                mask_valid_round_cummulative_all,
                mask_valid_round_window_all,
                feats_all,
                lbls_all,
                acc_flag_all
            )
        ],
        total_counts_per_round.view(-1, G)
    )):
        assert torch.all(gen_round[gather_idxs_per_round_cummulative] <= arange_gen_round.view(G, 1))
        acc_flags_per_th = {k : v.reshape(-1, N)[i] for k, v in acc_flags_per_th_all.items()}

        mask_valid_round_cummulative, gather_idxs_per_round_cummulative, mask_valid_window_simple = [
            t.unsqueeze(1).repeat(1, CV, 1) # [G, CV, N]
            for t in (mask_valid_round_cummulative, gather_idxs_per_round_cummulative, mask_valid_window_simple)
        ]

        var_to_hide = current_simulation_objects["var_to_hide"]
        F = dgp.F
        model_vars = [f for f in range(F) if f != var_to_hide]

        feats_per_round = feats[gather_idxs_per_round_cummulative][..., model_vars] # [G, CV, N, F]
        feats_per_round.masked_fill_(~mask_valid_round_cummulative.unsqueeze(-1), float('nan'))
        lbls_per_round = lbls[gather_idxs_per_round_cummulative]   # [G, CV, N]
        lbls_per_round.masked_fill_(~mask_valid_round_cummulative, float('nan'))
        accepts_per_round = {
            k : f[gather_idxs_per_round_cummulative] & mask_valid_round_cummulative
            for k, f in acc_flags_per_th.items()
        }

        rng = torch.Generator(device=credit_data.device).manual_seed(1807)

        perfect_bayes_classif = PerfectBayesClassifier(
            slice_dgp_across_first_dim(dgp, i), False, False, var_to_hide=var_to_hide
        )

        stats_perf_bayes = {}
        metric_funs = {
            "roc": batched_auroc,
            "ks" : lambda scores, targets, mask_valid : batched_ks_statistic(
                    scores, targets, mask_valid, return_thresholds=False
                )[0]
            }
        print("Beginning stats of i =", i)
        for exp in EXPECTATION_TYPES:
            print("\tBeginning exp:", exp)
            for m in METRIC_CATEGORIES:
                exp_mask = mask_valid_round_cummulative.clone()
                if exp == 'acc_based':
                    exp_mask &= acc_flags_per_th[exp + '_' + m]
                elif exp=='oracle_naive':
                    pass
                elif exp=='oracle_comparable':
                    exp_mask &= acc_flags_per_th['oracle_' + m]
                else:
                    raise AssertionError(f"Expectation type {exp} not recognized")
                
                feats = feats_per_round[:, 0].clone()
                lbls = lbls_per_round[:, 0].clone()
                mask_expectation = exp_mask[:, 0]

                batched_lr = BatchedLogistic(n_features=F-1, batch_shape = lbls.shape[:-1])
                batched_lr.fit(
                    X=feats,
                    y=lbls,
                    mask_valid_obs=mask_expectation
                )
                    
                th_is_oracle = exp.startswith("oracle")
                th = "oracle" if th_is_oracle else exp
                assert th in THRESHOLD_BASIS, "Threshold logic changed"
                assert False

                for perf in REAL_PERFORMANCE_TYPES:
                    real_mask = mask_valid_window_simple.clone()
                    perf_is_unb_future = perf=="unbiased_future"
                    if perf_is_unb_future:
                        pass
                    elif perf=='unbiased_acc':
                        real_mask &= acc_flags_per_th['oracle_' + m]
                    elif perf=='biased_acc':
                        real_mask &= acc_flags_per_th['acc_based_' + m]
                    else:
                        raise AssertionError("performance type not recognized")
                    
                    real_mask = real_mask[:, 0] # [G, N]

                    
                    if perf_is_unb_future:
                        real_lbl_stats = perf+ '_' + th
                        if not th_is_oracle:
                            # This difference is only for compatibility with the lines and colors pipeline
                            real_lbl_stats += "_" + m # Make sure based on same metric for acc-based
                        real_lbl_stats += '_real_' + m
                    else:
                        real_lbl_stats = perf+ '_' + th + "_" + m + '_real_' + m

                    B = (G + 1)//5
                    stats_perf_bayes[real_lbl_stats] = feats_per_round.new_full((G,), -1)
                    for start_g in range(0, G, B):
                        up_to_g = min(start_g + B, G)
                        prob_bad = perfect_bayes_classif.predict_prob_bad(feats[start_g:up_to_g])
                        stats_perf_bayes[real_lbl_stats][start_g:up_to_g] = metric_funs[m](prob_bad, lbls[start_g:up_to_g], mask_valid=real_mask[start_g:up_to_g])

                    stats_perf_bayes_all[real_lbl_stats].append(stats_perf_bayes[real_lbl_stats])

    print("writting off results\n")
    stats_perf_bayes_all = {k : torch.stack(v, dim=0) if dgp.is_batched else v[0] for k, v in stats_perf_bayes_all.items()}
    torch.save(stats_perf_bayes_all, os.path.join(sim_path_over_dir, path, "stats_perf_bayes.pt"))