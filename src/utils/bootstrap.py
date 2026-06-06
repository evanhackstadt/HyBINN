# Generated using Claude Sonnett 4.6

import numpy as np
from lifelines.utils import concordance_index


def bootstrap_cindex(times, events, risk_scores, n_bootstrap=1000, ci=0.95, seed=42):
    """
    Computes a bootstrap confidence interval for the C-index on a fixed set of predictions.
    Resamples patients with replacement and recomputes C-index on each resample.
    
    Call this after test evaluation, passing the stored predictions from evaluate().
    
    Args:
        times (array):       survival times for test patients
        events (array):      event indicators (1=died, 0=censored) for test patients
        risk_scores (array): predicted risk scores (risk_final) for test patients
        n_bootstrap (int):   number of bootstrap iterations (default 1000)
        ci (float):          confidence interval width (default 0.95 for 95% CI)
        seed (int):          seed for the bootstrap RNG (use the run seed for reproducibility)
    
    Returns:
        dict with keys: mean, std, lower, upper, n_bootstrap, ci_level, n_valid
    """
    rng = np.random.default_rng(seed)
    n = len(times)
    bootstrap_cindices = []
    n_skipped = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)   # resample with replacement
        t_boot = times[idx]
        e_boot = events[idx]
        r_boot = risk_scores[idx]

        # C-index is undefined if no events in the resample; skip rather than fail
        if e_boot.sum() == 0:
            n_skipped += 1
            continue

        try:
            c = concordance_index(t_boot, -r_boot, e_boot)
            bootstrap_cindices.append(c)
        except Exception:
            n_skipped += 1
            continue

    bootstrap_cindices = np.array(bootstrap_cindices)
    alpha = (1.0 - ci) / 2.0

    return {
        "mean":        float(np.mean(bootstrap_cindices)),
        "std":         float(np.std(bootstrap_cindices)),
        "lower":       float(np.percentile(bootstrap_cindices, 100 * alpha)),
        "upper":       float(np.percentile(bootstrap_cindices, 100 * (1.0 - alpha))),
        "n_bootstrap": n_bootstrap,
        "n_valid":     len(bootstrap_cindices),    # n_bootstrap minus any skipped resamples
        "ci_level":    ci,
    }