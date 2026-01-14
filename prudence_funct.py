import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hourly_rolling_cap(df, col, window="60D", q=0.99, min_periods=24*14):
    """
    Compute an hour-of-day conditional rolling cap (quantile threshold) for a price/revenue series.

    Financial intuition
    - Produces a time-varying “prudence threshold” used to limit spike-driven overstatement.
    - The cap is computed separately for each hour-of-day to respect intraday structure
      (e.g., morning/evening price regimes), rather than applying a single cap to all hours.

    Inputs
    - df : pd.DataFrame
        Must contain:
          1) a datetime-like index (or at least an index that supports rolling windows),
          2) the column `col` holding the raw values (e.g., EUR/MW),
          3) a column named "hour" with values 0–23 (typically df.index.hour).
        The function groups by df["hour"] and computes a rolling quantile within each hour bucket.
    - col : str
        Name of the column to cap (e.g., "FCR_capacity_EUR/MW").
    - window : str or offset-like (default "60D")
        Rolling lookback window used to estimate the cap from recent history.
        Example: "60D" means the last 60 days of observations *for that same hour-of-day*.
    - q : float (default 0.99)
        Quantile used as the cap (e.g., 0.99 means the 99th percentile).
        Higher q => less strict capping (cap sits higher).
    - min_periods : int (default 24*14)
        Minimum number of observations required within each hour bucket to define the cap.
        If insufficient history exists, cap values are NaN until this threshold is met.

    Output
    - pd.Series
        A time-indexed Series aligned to the original df index (after concat+sort),
        containing the rolling cap (same units as df[col], e.g., EUR/MW).
        Cap is shifted by 1 timestep so that the cap at time t is based only on information
        available up to t-1 (avoids look-ahead bias).

    Notes
    - Look-ahead prevention: `.shift(1)` ensures the cap uses past data only.
    - Missing data: if df[col] has NaNs, the rolling quantile behaves accordingly.
    - The returned Series may have NaNs early in the sample or for sparse hours.
    """
    caps = []
    for h, g in df[[col]].assign(hour=df["hour"]).groupby("hour"):
        s = g[col]
        cap_h = s.rolling(window=window, min_periods=min_periods).quantile(q).shift(1)
        caps.append(cap_h)
    return pd.concat(caps).sort_index()


def apply_soft_cap(s: pd.Series, cap: pd.Series, alpha: float = 0.25) -> pd.Series:
    """
    Apply a soft cap to a series to reduce extreme upside spikes while preserving continuity.

    Purpose (financial intuition)
    - This is a “prudence haircut” mechanism: it reduces revenue/price values above a
      data-driven threshold (cap), rather than hard-clipping them.
    - Soft capping keeps some upside participation but dampens tail events that can
      distort expected revenues and risk metrics (useful for bankability).

    Mechanics
    - If cap is NaN: keep the raw value s (no prudence applied where cap is undefined).
    - If s is NaN: keep NaN (missingness is preserved).
    - If s <= cap: keep s unchanged.
    - If s > cap: compress the exceedance above the cap:
          prudent = cap + alpha * (s - cap)

    Inputs
    - s : pd.Series
        Raw series to be capped (e.g., EUR/MW). Must align (by index) with `cap`.
    - cap : pd.Series
        Cap series (same units as s). Typically output of hourly_rolling_cap.
    - alpha : float (default 0.25)
        Softness / severity parameter applied to exceedances:
        - alpha = 0   -> values above cap are set to cap (hard cap)
        - alpha = 1   -> no capping effect (prudent equals raw)
        - 0 < alpha < 1 -> partial reduction (typical prudence setting)

    Output
    - pd.Series
        Prudence-adjusted series (same index and units as s).

    Notes
    - “Never turns valid observations into NaN”:
      if s[t] is defined and cap[t] is missing, s[t] is retained.
    - This transform is monotonic in s (preserves ranking within exceedances).
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
    """
    Visualize raw vs prudent series together with the rolling cap threshold.

    Purpose (auditability)
    - Provides a transparent view of:
        (i) the original revenue stream (raw),
       (ii) the prudence-adjusted stream (prudent),
      (iii) the cap level used to control spikes (cap),
      and highlights where a prudence haircut is applied.
    - The plot title reports data quality and prudence binding frequency.

    Inputs
    - df : pd.DataFrame
        Must contain columns raw_col, prudent_col, cap_col and a datetime-like index.
    - raw_col : str
        Column name for the raw series (e.g., "FCR_capacity_EUR/MW").
    - prudent_col : str
        Column name for the prudent series (output of apply_soft_cap).
    - cap_col : str
        Column name for the cap series (output of hourly_rolling_cap).
    - title : str
        Plot title prefix (will be augmented with missingness and capped share).
    - start, end : optional
        Index slice boundaries passed to df.loc[start:end].
    - resample : str or None
        If not None, resample data to a coarser frequency using mean (e.g., "D" for daily)
        to improve readability and reduce visual noise.
    - alpha_raw, alpha_prudent, alpha_cap_band : float
        Transparency for raw line, prudent line, and cap band fill respectively.
    - lw_raw, lw_prudent : float
        Line widths for raw and prudent series.

    Output
    - None (produces a matplotlib plot)

    Reported diagnostics in title
    - missing(raw): share of timestamps where raw is missing (percentage)
    - capped share: share of timestamps where cap is defined and raw > cap (percentage)

    Notes
    - The cap is shown as a filled band from 0 up to cap. This emphasizes the “allowed”
      region under the prudence threshold.
    - The “haircut” region (raw - prudent) is shaded where prudent < raw.
    """
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

    # Show where prudent < raw
    haircut_mask = raw.notna() & pru.notna() & (raw > pru)
    ax.fill_between(d.index, pru, raw, where=haircut_mask, alpha=0.15, zorder=2)

    ax.set_title(f"{title} | missing(raw)={missing_pct:.1f}% | capped share={cap_share:.1f}%")
    ax.set_ylabel("EUR/MW")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def stationary_bootstrap_indices(n, p, rng):
    """
    Generate index positions for the Stationary Bootstrap (Politis & Romano).

    Purpose (uncertainty modelling)
    - Creates resampled index sequences that preserve time-dependence by sampling
      contiguous blocks, but with *random* block lengths.
    - Block lengths follow a geometric distribution governed by p, making the bootstrap
      “stationary” (no fixed block boundaries) and suitable for autocorrelated series.

    Inputs
    - n : int
        Length of the time series to resample.
    - p : float
        Probability of starting a new block at each step.
        Expected block length = 1/p.
        Example: exp_block_len = 48 hours -> p = 1/48.
    - rng : np.random.Generator
        Random number generator instance for reproducibility.

    Output
    - np.ndarray of shape (n,)
        Integer indices in [0, n-1] representing the resampling path.

    Mechanics
    - Start at a random index.
    - For each subsequent time step:
        - with probability p: jump to a new random index (start a new block),
        - otherwise: move forward by 1 (wrap around at n).
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
    """
    Simulate revenue scenarios using the Stationary Bootstrap.

    Purpose (financial risk metrics)
    - Generates many plausible alternative histories consistent with the autocorrelation
      structure of the observed revenue series.
    - Used to estimate distributions of annual totals (P50/P90 style outcomes)
      without assuming normality or independent observations.

    Inputs
    - series : pd.Series
        1D time series of revenue/price observations (e.g., hourly EUR/MW),
        ideally with NaNs removed beforehand. The function uses series.to_numpy().
    - n_sims : int (default 2000)
        Number of simulated scenarios to generate.
    - exp_block_len : int (default 48)
        Expected block length in time steps (hours if series is hourly).
        Controls how much persistence is preserved:
          - larger => longer blocks => more autocorrelation retained
          - smaller => shorter blocks => more mixing / less persistence
        Internally p = 1/exp_block_len.
    - seed : int (default 7)
        Random seed for reproducibility.

    Output
    - np.ndarray of shape (n_sims, n)
        Each row is one simulated scenario path of length n (same units as input series).

    Notes
    - This function returns simulated *paths*, not aggregated totals.
      Annual totals are typically computed downstream by summing along axis=1.
    """
    rng = np.random.default_rng(seed)
    n = len(series)
    p = 1.0 / exp_block_len
    sims = np.empty((n_sims, n))
    x = series.to_numpy()
    for i in range(n_sims):
        idx = stationary_bootstrap_indices(n, p, rng)
        sims[i, :] = x[idx]
    return sims

def capping_stats(raw, cap):
    """
    Compute simple diagnostics on how often a cap is defined and how often it binds.

    Purpose (model governance / transparency)
    - Quantifies the “activity” of prudence:
        - Is the cap available most of the time (enough history)?
        - How frequently does raw exceed cap?
        - How large are exceedances when they occur?
    - These metrics help justify prudence settings (window, q, alpha) and document
      the impact of spike control on revenue projections.

    Inputs
    - raw : pd.Series
        Raw series (e.g., EUR/MW).
    - cap : pd.Series
        Cap series (e.g., EUR/MW), aligned by index with raw.

    Output
    - dict with:
        - cap_defined_share : float
            Share of all timestamps where cap is not NaN.
        - exceed_count : int
            Number of timestamps where both raw and cap are defined and raw > cap.
        - exceed_share_of_defined : float
            Fraction of defined timestamps (raw and cap present) where cap binds.
        - avg_exceedance : float
            Average exceedance magnitude (raw - cap) conditional on exceedance.
            Returns 0.0 if there are no exceedances.

    Notes
    - exceed_share_of_defined uses max(m.sum(), 1) to avoid division by zero.
    """
    m = raw.notna() & cap.notna()
    exceed = m & (raw > cap)
    return {
        "cap_defined_share": cap.notna().mean(),
        "exceed_count": int(exceed.sum()),
        "exceed_share_of_defined": float(exceed.sum() / max(m.sum(), 1)),
        "avg_exceedance": float((raw[exceed] - cap[exceed]).mean()) if exceed.any() else 0.0
    }


