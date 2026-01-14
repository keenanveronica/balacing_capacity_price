import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hourly_rolling_cap(df, col, window="60D", q=0.99, min_periods=24*14):
    caps = []
    for h, g in df[[col]].assign(hour=df["hour"]).groupby("hour"):
        s = g[col]
        cap_h = s.rolling(window=window, min_periods=min_periods).quantile(q).shift(1)
        caps.append(cap_h)
    return pd.concat(caps).sort_index()


def apply_soft_cap(s: pd.Series, cap: pd.Series, alpha: float = 0.25) -> pd.Series:
    """
    Soft cap that never turns valid observations into NaN.
    - If cap is NaN: keep raw s
    - If s is NaN: keep NaN
    - If s > cap: compress exceedance by alpha
    """
    y = s.copy()
    m = s.notna() & cap.notna()
    exceed = m & (s > cap)
    y.loc[exceed] = cap.loc[exceed] + alpha * (s.loc[exceed] - cap.loc[exceed])
    return y


def plot_prudence_component(
    df, raw_col, prudent_col, cap_col, title,
    start=None, end=None, resample=None,
    alpha_raw=1.0, alpha_prudent=0.55, alpha_cap_band=0.10,
    lw_raw=1.6, lw_prudent=1.6
):
    d = df.loc[start:end, [raw_col, prudent_col, cap_col]].copy()

    if resample is not None:
        d = d.resample(resample).mean()

    raw = d[raw_col]
    pru = d[prudent_col]
    cap = d[cap_col]

    valid_raw = raw.notna().sum()
    missing_pct = 100 * (1 - valid_raw / len(raw)) if len(raw) else 0

    capped_mask = raw.notna() & cap.notna() & (raw > cap)
    cap_share = 100 * capped_mask.sum() / max((raw.notna() & cap.notna()).sum(), 1)

    fig, ax = plt.subplots(figsize=(14, 4))

    # Cap as a band from y=0 up to the cap
    cap_mask = cap.notna()
    ax.fill_between(
        d.index, 0, cap,
        where=cap_mask,
        alpha=alpha_cap_band,
        label="rolling cap band",
        zorder=1
    )

    # Raw + prudent lines on top
    ax.plot(d.index, raw, label="raw", alpha=alpha_raw, linewidth=lw_raw, zorder=3)
    ax.plot(d.index, pru, label="prudent", alpha=alpha_prudent, linewidth=lw_prudent, zorder=4)

    # Optional: show prudence haircut (where prudent < raw)
    haircut_mask = raw.notna() & pru.notna() & (raw > pru)
    ax.fill_between(d.index, pru, raw, where=haircut_mask, alpha=0.15, zorder=2)

    ax.set_title(f"{title} | missing(raw)={missing_pct:.1f}% | capped share={cap_share:.1f}%")
    ax.set_ylabel("EUR/MW")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def stationary_bootstrap_indices(n, p, rng):
    """
    Stationary bootstrap (Politis & Romano): blocks with geometric length.
    p: probability of starting a new block; expected block length = 1/p
    """
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t-1] + 1) % n
    return idx


def revenue_scenarios_stationary_bootstrap(series, n_sims=2000, exp_block_len=48, seed=7):
    rng = np.random.default_rng(seed)
    n = len(series)
    p = 1.0 / exp_block_len
    sims = np.empty((n_sims, n))
    x = series.to_numpy()
    for i in range(n_sims):
        idx = stationary_bootstrap_indices(n, p, rng)
        sims[i, :] = x[idx]
    return sims
