# Balancing Capacity Price – Prudential Revenue Estimation

This repository contains tools to model balancing capacity revenues in the Dutch electricity market.

The framework is designed to:

- Construct historical capacity price series
- Apply prudence adjustments (soft caps)
- Quantify tail risk exposure
- Simulate forward revenue distributions using stationary bootstrap
- Produce P50 / P90 style metrics for investment cases

---

## 1. Project Objective

Balancing capacity markets (i.e. FCR, aFRR) exhibit:

- Intraday seasonality
- Heavy right tails
- Structural regime shifts
- Autocorrelation
- Extreme spike events

Raw historical averages systematically **overstate bankable revenue**.

This repository implements a prudential framework that:

1. Applies hour-conditional rolling quantile caps
2. Soft-compresses tail events
3. Preserves autocorrelation via stationary bootstrap
4. Produces revenue scenario distributions

The result is a structured, downside-aware revenue model.

---

## 2. Repository Structure

balancing_capacity_price/
│
├── prudence_funct.py
├── best_case_balancingUP_revenue.ipynb
├── best_case_balancingDOWN_revenue.ipynb
├── capacity_balancing_NL.ipynb
├── capacity_aFRR_NL.ipynb
├── activated_aFRR_NL.ipynb
└── README.md

---

## 3. Core Module: prudence_funct.py

### 3.1 Hourly Rolling Quantile Cap

`hourly_rolling_cap()`

Computes an hour-of-day conditional rolling quantile threshold:

- Rolling lookback window (e.g., 60 days)
- Computed separately per hour-of-day
- Shifted to avoid look-ahead bias
- Controls spike exposure while respecting intraday seasonality

Financial interpretation:
A time-varying prudence threshold consistent with lender-style stress assumptions.

---

### 3.2 Soft Cap Mechanism

`apply_soft_cap()`

Instead of hard clipping:

prudent = cap + α (raw − cap)

Where:

- α = 0 → hard cap
- α = 1 → no prudence
- 0 < α < 1 → partial tail participation

This ensures:

- Continuity
- Realistic upside retention
- Controlled tail exposure
- Improved bankability

---

### 3.3 Stationary Bootstrap

`revenue_scenarios_stationary_bootstrap()`

Implements Politis & Romano stationary bootstrap:

- Preserves autocorrelation
- Retains regime persistence
- Does not assume normality
- Suitable for heavy-tailed markets

Used to simulate:

- Annual revenue distributions
- P50 / P90 outcomes
- Downside risk metrics

---

### 3.4 Prudence Diagnostics

`capping_stats()`

Outputs:

- Share of timestamps where cap is defined
- Frequency of binding
- Average exceedance magnitude

Useful for:

- Lender transparency
- Investment committee documentation
- Risk appendix reporting

---

## 4. Methodology

### Step 1 — Raw Historical Series

Obtain:

- Capacity prices (EUR/MW)
- Hourly timestamp index
- Hour-of-day column

---

### Step 2 — Construct Prudence Cap

Typical parameters:

- Rolling window: 60 days
- Quantile: 99%
- Minimum history requirement

Produces a dynamic spike threshold.

---

### Step 3 — Apply Soft Compression

Select α (e.g. 0.25).

This reduces extreme spikes while preserving structure.

---

### Step 4 — Simulate Revenue Scenarios

Stationary bootstrap:

- Expected block length (e.g. 48 hours)
- 2,000+ simulations
- Aggregate annual totals

---

### Step 5 — Extract Risk Metrics

Compute:

- Mean
- Median (P50)
- P90
- P95 downside
- CVaR (optional)

---

## 5. Data Sources

- FCR capacity data used in this project has been sourced from the official Transparency Platform of Regelleistung (https://www.regelleistung.net/en-us/Data/Datacenter).
  The datasets were downloaded manually on a monthly basis.

- All remaining datasets (e.g., aFRR, activation volumes, and related balancing data) were imported via the ENTSO-E Transparency Platform API using a personal access token.

Users replicating this work must obtain their own API token from ENTSO-E and independently download any required Regelleistung datasets.

---

## 6. Dependencies
This repository includes a fully specified `environment.yml` file to ensure reproducibility.

To recreate the environment:

```bash
conda env create -f environment.yml
conda activate balancing-capacity-env
```
---

## 7. Key Assumptions

- Historical structure is informative of future regimes
- Autocorrelation matters
- Tail spikes cannot not be fully banked
- Revenue modelling must be prudence-adjusted

---

## 8. Limitations

This framework:

- Does not incorporate forward curve information
- Does not model structural market reform scenarios
- Assumes stationary dependence within bootstrap window

Further structural modelling may be required for long-dated projections.

---

## 9. Disclaimer

This repository provides a preliminary framework for revenue modelling intended for experimenting and educational purposes.

It does not constitute investment advice.

All market data remains the property of its respective providers.
Users are responsible for ensuring compliance with the terms of use of Regelleistung and ENTSO-E Transparency Platform data.

