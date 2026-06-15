"""Sleep/wake conditional probability + next-tweet prediction — v4.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import beta as beta_dist
from scipy.stats import norm as norm_dist
from scipy.stats import genpareto
import pytz
import sys
import os
import csv
import time

# ── Configuration ────────────────────────────────────────────────────────────
_DATA           = os.environ.get("DATA_DIR", ".")
CSV_FILENAME    = os.path.join(_DATA, "elonmusk_tweet_history.csv")
OUTPUT_FILENAME = os.path.join(_DATA, "final_output_v4.txt")

SLEEP_GAP_HOURS      = 3.0
EARLIEST_BEDTIME_H   = 22
LATEST_BEDTIME_H     = 31
EARLIEST_WAKE_H      = 8
LATEST_WAKE_H        = 13
BEDTIME_SESSION_MIN  = 90
MORNING_SESSION_MIN  = 90
ANOMALY_GAP_MULT     = 2.5
EST                  = pytz.timezone("America/New_York")

# ── Sleep-State Inference Configuration ──────────────────────────────────────
SLEEP_INFER_NOW        = None    # "6/12/2026 02:30" (EST) or None → current clock time
SLEEP_INFER_LAST_TWEET = None    # "6/12/2026 01:10" (EST) or None → most recent tweet in CSV
SHRINK_KAPPA           = 8.0     # fallback shrinkage strength; v2 tunes kappa by CV (#8)
KAPPA_GRID             = [2.0, 4.0, 8.0, 16.0, 32.0]
SCENARIO_LAST_HOURS    = [22, 23, 24, 25, 26, 27]   # last-tweet clock: 10PM..3AM (session scale)
SCENARIO_SILENCE_MIN   = [30, 60, 90, 120, 180, 240, 300, 360]
CONFIRM_TARGETS        = [0.80, 0.90, 0.95]
CONFIRM_SEARCH_MAX_MIN = 480     # search silence thresholds up to 8 h
DIRECT_MATCH_TOL_H     = 0.75    # ±45 min last-tweet-time tolerance for empirical cross-check
NIGHT_END_SESSION_H    = 29      # next tweet at/after 5 AM (session hour 29) ends the night
MAX_WAKE_SESSION_H     = 38      # ignore multi-day disappearances when modelling wake times

# v2 evidence matching (#2): centered kernel on the last-tweet clock time
# instead of floor-hour bins, widened until the pool is adequately populated.
EVIDENCE_S0_TOL_H      = 0.75            # base kernel half-width (hours)
EVIDENCE_TOL_WIDEN     = [0.75, 1.5, 3.0]
WAKE_BEDTIME_TOL_H     = 1.5             # (#3) wake distribution conditioned on bedtime ± this
WEEKDAY_POOL_WEIGHT    = 2.0             # (#5) same-weekday rows up-weighted in quantile pools
TIER_POOL_WEIGHT       = 1.5             # (#5) matching-activity-tier rows up-weighted
PRED_QUANTILES         = [0.25, 0.50, 0.75, 0.90]

# ── Monte Carlo Next-Tweet Configuration (v3) ────────────────────────────────
# The next-tweet time carries two distinct uncertainties; the simulation keeps
# them separate. ALEATORIC: even with the model fixed, the time is random — draw
# it from the evidence-matched gap pool, each draw perturbed by Gaussian timing
# jitter (MC_JITTER_MIN) so the discrete pool becomes a smooth continuous law.
# EPISTEMIC: the pool is finite, so the law itself is uncertain — a weighted
# bootstrap (MC_BOOTSTRAP resamples) yields a credible band on every reported
# probability.
MC_N_SAMPLES           = 50000   # aleatoric predictive draws
MC_BOOTSTRAP           = 400     # epistemic (finite-pool) bootstrap resamples
MC_JITTER_MIN          = 20.0    # timing-uncertainty sd added per draw (minutes)
MC_BUCKET_MAX_H        = 14      # hour-by-hour profile horizon (hours from now)
MC_SEED                = 12345   # reproducible simulation
MC_DAYTIME_MIN_POOL    = 8       # min awake gaps before the daytime predictor falls back

# ── Extreme-Silence Tail Model (v4 — EVT / POT-GPD) ──────────────────────────
# Inter-tweet gaps are deseasonalized by the circadian intensity μ(hour) (random
# time-change): the rescaled gap = ∫ μ dt over the gap = "expected tweets missed".
# A long overnight quiet accrues little (low μ at night) → not anomalous; a long
# DAYTIME silence accrues a lot → anomalous. A Peaks-Over-Threshold GPD tail is
# fit to the rescaled gaps; threshold-stability gives the residual-life law for a
# silence already past the support, mapped back to wall-clock via inverse μ.
TAIL_THRESHOLD_Q       = 0.95    # POT threshold for FITTING the GPD (quantile of rescaled gaps)
TAIL_MIN_EXCEEDANCES   = 30      # need at least this many exceedances to fit the GPD
# The tail regime ACTIVATES (preempting the night-rest model) only when the
# silence is both deseasonalized-unusual (rescaled elapsed > the fit threshold)
# AND the last tweet can no longer be conditioned on — i.e. it has run past the
# wake window OR the direct evidence pool has fewer than this many gaps. This
# stops a routine peak-hour quiet (top-5% rescaled ≈ 2 h at 1 AM) from being
# mislabelled "extreme" and bypassing the P(asleep) inference.
TAIL_MIN_CONDITION_POOL = 3
TAIL_XI_GRID           = (-0.4, 0.8, 25)   # GPD shape ξ posterior grid (min, max, n)
TAIL_SIGMA_GRID_MULT   = (0.15, 3.5, 25)   # GPD scale grid, as multiples of exceedance mean
TAIL_POSTERIOR_DRAWS   = 400     # draws from the Bayesian (grid) GPD posterior
TAIL_SAMPLES           = 20000   # next-tweet predictive draws in the tail regime
WEEKDAY_TAIL_WEIGHT    = 2.0     # partial pooling: up-weight same-weekday exceedances
TAIL_INVERT_HORIZON_H  = 24 * 5  # wall-clock horizon for inverse-intensity lookup
REGIME_CHANGE_PROB_MAX = 0.35    # max competing-risk "dormant" mixing weight beyond the max
NEXT_TWEET_WINDOWS_MIN = [15, 30, 60]   # ± windows for "prob the tweet lands at the target"
NEXT_TWEET_TARGET      = None    # None → use the predicted median; or "6/13/2026 21:30" (EST)

# Live silence is measured from the CSV's last row; if the updater polls every
# N minutes, observed silence carries a +N/2 ingestion-lag bias. Set this to
# the updater's mean latency (minutes) to subtract it from live silences.
INGEST_LAG_MIN         = 0.0

# Sleep zone (used for the regime-matched base rate and display; v2 no longer
# pins P(asleep)=0 outside it — see #6).
SLEEP_ZONE_WAKE_H      = 7                            # sleep zone ends at 7 AM EST
SLEEP_ZONE_START_S     = float(EARLIEST_BEDTIME_H)   # 22.0 → 10 PM EST
SLEEP_ZONE_END_S       = float(SLEEP_ZONE_WAKE_H + 24)   # 31.0 → 7 AM EST next day

# "Night is over" cutoff for the live readout (P(asleep)=0 override and the
# next-tweet prediction switch). This is the model's LATEST plausible wake, not
# the 7 AM display sleep-zone end: he sometimes sleeps until ~1 PM, so the
# night-rest model (and its morning-wake prediction) stays valid until then.
# Tying the cutoff to the 7 AM zone instead caused a ~2.5h discontinuity at the
# boundary and discarded the legitimate 7 AM–1 PM morning-wake prediction.
NIGHT_OVER_SESSION_H   = float(LATEST_WAKE_H + 24)   # 37.0 → 1 PM EST next day

# ── Monitor Mode Configuration ───────────────────────────────────────────────
MONITOR_INTERVAL_MIN   = 1
MONITOR_LOG_FILENAME   = os.path.join(_DATA, "next_tweet_monitor_log_v4.csv")
MONITOR_MAX_QUEUED     = 500
DAILY_REPORT_HOUR_EST  = 22      # retained for reference; no longer schedules the report
REPORT_INTERVAL_MIN    = 5       # re-write OUTPUT_FILENAME this often (live recalculation)

# ── Activity Covariate Configuration ─────────────────────────────────────────
ACTIVITY_WINDOWS_H     = [1, 2, 3, 4, 6, 8, 12, 24]
ACTIVITY_TIER_LABELS   = ["LOW", "MID", "HIGH"]
ACTIVITY_CV_FOLDS      = 5
ACTIVITY_MIN_BIN_N     = 30
ACTIVITY_MIN_FOLDS_POS = 4    # (#7) adoption also needs >= this many positive folds

# ── Launch Day Timestamps ─────────────────────────────────────────────────────
LAUNCH_TIMESTAMPS_UTC = [
    "2026-04-14T09:23:00Z", "2026-04-11T11:41:00Z", "2026-04-11T05:04:00Z",
    "2026-04-07T02:50:00Z", "2026-04-02T11:55:00Z", "2026-03-30T23:15:00Z",
    "2026-03-30T11:02:00Z", "2026-03-26T23:03:00Z", "2026-03-22T14:47:00Z",
    "2026-03-20T21:51:00Z", "2026-03-19T14:20:00Z", "2026-03-17T13:27:00Z",
    "2026-03-17T05:19:00Z", "2026-03-14T12:37:00Z", "2026-03-13T14:57:00Z",
    "2026-03-10T04:19:00Z", "2026-03-08T15:00:00Z", "2026-03-04T10:52:00Z",
    "2026-03-02T02:56:00Z", "2026-03-01T10:10:00Z", "2026-02-27T12:16:00Z",
    "2026-02-25T14:17:00Z", "2026-02-24T23:04:00Z", "2026-02-22T03:47:00Z",
    "2026-02-21T09:04:00Z", "2026-02-20T01:41:00Z", "2026-02-16T07:59:00Z",
    "2026-02-15T01:59:00Z", "2026-02-11T17:11:00Z", "2026-02-07T20:58:00Z",
    "2026-02-02T15:47:00Z", "2026-01-30T07:22:00Z", "2026-01-29T17:53:00Z",
    "2026-01-28T04:53:00Z", "2026-01-25T17:30:00Z", "2026-01-22T05:47:00Z",
    "2026-01-18T23:31:00Z", "2026-01-17T04:39:00Z", "2026-01-14T18:08:00Z",
    "2026-01-12T21:08:00Z", "2026-01-11T13:44:00Z", "2026-01-09T21:41:00Z",
    "2026-01-04T06:48:00Z", "2026-01-03T02:09:00Z"
]

DAYS_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_time(dt):
    if dt is None or (isinstance(dt, float) and np.isnan(dt)):
        return "N/A"
    return dt.strftime("%I:%M %p EST")

def decimal_to_time_str(decimal_hours):
    if np.isnan(decimal_hours):
        return "N/A"
    decimal_hours = decimal_hours % 24
    h = int(decimal_hours)
    m = int(round((decimal_hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    h = h % 24
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12:02d}:{m:02d} {ampm} EST"

def circular_mean_hours(hours_list):
    hours_list = [h for h in hours_list if not np.isnan(h)]
    if not hours_list:
        return np.nan
    angles = [h / 24.0 * 2 * np.pi for h in hours_list]
    sin_mean = np.mean([np.sin(a) for a in angles])
    cos_mean = np.mean([np.cos(a) for a in angles])
    mean_angle = np.arctan2(sin_mean, cos_mean)
    mean_hours = mean_angle / (2 * np.pi) * 24
    if mean_hours < 0:
        mean_hours += 24
    return mean_hours

def night_hour(dt):
    h = dt.hour + dt.minute / 60.0
    return h + 24 if h < 22 else h

def parse_hhmm_or_datetime(s):
    s = s.strip()
    if "/" in s:
        dt = pd.to_datetime(s)
        return dt.date(), dt.hour, dt.minute
    else:
        h, m = map(int, s.split(":"))
        return None, h, m

def _logit(p):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return float(np.log(p / (1.0 - p)))

def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))

def _weighted_quantiles(values, weights, qs):
    """Weighted empirical quantiles (midpoint rule)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        return [np.nan] * len(qs)
    order = np.argsort(values)
    v, w = values[order], np.maximum(weights[order], 1e-9)
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / cw[-1]
    return [float(np.interp(q, cw, v)) for q in qs]

# ── Core Data Functions ─────────────────────────────────────────────────────
def load_data(filename):
    df = pd.read_csv(filename)
    df.columns = [c.strip() for c in df.columns]
    dt_col  = [c for c in df.columns if "datetime" in c.lower() or "date" in c.lower()][0]
    cnt_col = [c for c in df.columns if any(x in c.lower() for x in ["cumulative","count","tweet"])][0]

    df = df.rename(columns={dt_col: "DateTime_UTC", cnt_col: "Cumulative_Tweet_Count"})
    df["DateTime_UTC"] = pd.to_datetime(df["DateTime_UTC"], utc=True)
    df = df.sort_values("DateTime_UTC").reset_index(drop=True)

    df["Tweet_Count"] = df["Cumulative_Tweet_Count"].diff().fillna(df["Cumulative_Tweet_Count"].iloc[0])
    df["Tweet_Count"] = df["Tweet_Count"].clip(lower=0).astype(int)
    df["DateTime_EST"] = df["DateTime_UTC"].dt.tz_convert(EST)
    df = df[df["Tweet_Count"] > 0].reset_index(drop=True)
    return df

def expand_to_tweet_events(df):
    events = []
    for _, row in df.iterrows():
        events.extend([row["DateTime_EST"]] * int(row["Tweet_Count"]))
    return pd.Series(events).sort_values().reset_index(drop=True)

def find_sleep_periods(tweet_times):
    if len(tweet_times) < 2:
        return []
    n = len(tweet_times)
    sleep_periods = []
    gaps = [(tweet_times[i+1] - tweet_times[i]).total_seconds() / 3600.0 for i in range(n-1)]
    median_gap = np.median(gaps) if gaps else 3.0

    i = 0
    while i < n - 1:
        g = gaps[i]
        if g >= SLEEP_GAP_HOURS:
            t0 = tweet_times[i]
            t1 = tweet_times[i + 1]
            nh0 = night_hour(t0)
            h1  = t1.hour + t1.minute / 60.0

            if (EARLIEST_BEDTIME_H <= nh0 <= LATEST_BEDTIME_H and
                EARLIEST_WAKE_H <= h1 <= LATEST_WAKE_H):
                if not (g > 20 and g > ANOMALY_GAP_MULT * max(median_gap, 1.0)):
                    day_label = (t0 - timedelta(days=1)).date() if nh0 >= 24 else t0.date()
                    sleep_periods.append({
                        "sleep_start": t0,
                        "wake_time": t1,
                        "gap_hours": g,
                        "date_label": day_label,
                        "weekday": day_label.strftime("%A"),
                    })
            i += 1
        else:
            i += 1
    return sleep_periods

def count_session_tweets(tweet_times, anchor_time, window_minutes, direction="before"):
    if direction == "before":
        start = anchor_time - timedelta(minutes=window_minutes)
        mask = (tweet_times >= start) & (tweet_times <= anchor_time)
    else:
        end = anchor_time + timedelta(minutes=window_minutes)
        mask = (tweet_times >= anchor_time) & (tweet_times <= end)
    return int(mask.sum())

def get_launch_dates_est():
    launch_dates = set()
    for ts in LAUNCH_TIMESTAMPS_UTC:
        dt_utc = pd.to_datetime(ts, utc=True)
        dt_est = dt_utc.tz_convert(EST)
        launch_dates.add(dt_est.date())
    return launch_dates

# ── Sleep Summary ────────────────────────────────────────────────────────────
def sleep_analysis_subset(sp_df, label, file):
    print("\n" + "=" * 70, file=file)
    print(f"  PER-WEEKDAY SLEEP SUMMARY [{label}]", file=file)
    print("=" * 70, file=file)

    for day in DAYS_ORDER:
        day_df = sp_df[sp_df["weekday"] == day].copy()
        if day_df.empty:
            continue

        n = len(day_df)

        night_sleep_hrs = [night_hour(r["sleep_start"]) for _, r in day_df.iterrows()]
        avg_sleep_nh = circular_mean_hours(night_sleep_hrs)
        avg_sleep_str = decimal_to_time_str(avg_sleep_nh)

        wake_hrs = [r["wake_time"].hour + r["wake_time"].minute / 60.0 for _, r in day_df.iterrows()]
        avg_wake_h = circular_mean_hours(wake_hrs)
        avg_wake_str = decimal_to_time_str(avg_wake_h)

        day_df = day_df.copy()
        day_df["night_sleep_hr"] = [night_hour(r["sleep_start"]) for _, r in day_df.iterrows()]
        day_df["wake_hr_only"] = day_df["wake_time"].apply(lambda t: t.hour + t.minute / 60.0)

        earliest_sleep = day_df.loc[day_df["night_sleep_hr"].idxmin(), "sleep_start"]
        latest_sleep   = day_df.loc[day_df["night_sleep_hr"].idxmax(), "sleep_start"]
        earliest_wake  = day_df.loc[day_df["wake_hr_only"].idxmin(), "wake_time"]
        latest_wake    = day_df.loc[day_df["wake_hr_only"].idxmax(), "wake_time"]

        avg_sleep_dur = day_df["gap_hours"].mean()
        min_sleep_dur = day_df["gap_hours"].min()
        max_sleep_dur = day_df["gap_hours"].max()

        bt = day_df["bedtime_tweets"]
        mt = day_df["morning_tweets"]
        p_bed_lt5 = (bt < 5).mean()
        p_morn_lt5 = (mt < 5).mean()

        print(f"\n  ┌─ {day.upper()} ({n} nights) ────────────────────────────────────────", file=file)
        print(f"  │ Avg bedtime : {avg_sleep_str} | Avg wake-up : {avg_wake_str}", file=file)
        print(f"  │ Earliest bed : {fmt_time(earliest_sleep)} ({earliest_sleep.strftime('%Y-%m-%d')}) | "
              f"Latest bed : {fmt_time(latest_sleep)} ({latest_sleep.strftime('%Y-%m-%d')})", file=file)
        print(f"  │ Earliest wake : {fmt_time(earliest_wake)} ({earliest_wake.strftime('%Y-%m-%d')}) | "
              f"Latest wake : {fmt_time(latest_wake)} ({latest_wake.strftime('%Y-%m-%d')})", file=file)
        print(f"  │ Sleep duration: avg={avg_sleep_dur:.1f}h min={min_sleep_dur:.1f}h max={max_sleep_dur:.1f}h", file=file)
        print(f"  │ Bed session : avg={bt.mean():.1f} tweets P(<5)={p_bed_lt5:.1%}", file=file)
        print(f"  │ Morn session : avg={mt.mean():.1f} tweets P(<5)={p_morn_lt5:.1%}", file=file)
        print(f"  └────────────────────────────────────────────────────────", file=file)


# ── Sleep-State Inference (v2) ───────────────────────────────────────────────
#
# Evidence at query time τ: last tweet at τ0 = τ − g, silent for g hours.
# Every historical inter-tweet gap is one trial. Classes (#1) — all three are
# OBSERVABLE (no label noise):
#   terminal   — gap >= SLEEP_GAP_HOURS and ends at/after NIGHT_END_SESSION_H
#                (5 AM): the silence demonstrably lasted until morning.
#   shortnight — gap >= SLEEP_GAP_HOURS starting in the bedtime window but
#                ending BEFORE 5 AM: a short night sleep that ended early.
#   awake      — everything else.
#
# Two posteriors are estimated from the same evidence pool (#1):
#   p_until_morning — P(terminal): the betting-relevant "no more tweets until
#                     5 AM" event; shortnight gaps are real negatives here.
#   p_asleep        — P(terminal or shortnight): "currently in a night-rest
#                     gap"; the sleep-state interpretation.
# Estimation is by direct conditioning (#2): among gaps that started within a
# ±EVIDENCE_S0_TOL_H kernel of τ0's clock time AND lasted at least g, the
# (kernel-weighted) positive share. Shrinkage chain: regime base rate →
# kernel pool → activity tier, each with κ pseudo-counts (κ tuned by grouped
# CV, #8). The weekday enters as a log-odds main effect estimated from all
# gaps on the same regime side (#8), and the credible band is the Beta
# posterior implied by the shrinkage (#4).
#
# The predictive distribution of the next-tweet time is read directly off the
# end times of the evidence-matched gaps (weighted quantiles, #5); the
# two-branch mixture expectation is kept as a secondary diagnostic with the
# wake branch conditioned on bedtime proximity (#3).
#
# Session-hour scale: anchored at 1 PM EST so one night (13:00 → 12:59 next
# day) is contiguous: 22 = 10 PM, 24 = midnight, 26 = 2 AM, 33 = 9 AM.
# Kernels do not wrap across the 1 PM boundary (negligible: lowest-stakes hour
# of the cycle). end_s = s0 + elapsed drifts ±1h from clock time on the two
# DST nights per year — accepted.

def session_hour(dt):
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    return h if h >= 13 else h + 24

def night_anchor_date(dt):
    return dt.date() if dt.hour >= 13 else (dt - timedelta(days=1)).date()

def jeffreys_interval(k, n, level=0.95):
    if n == 0:
        return 0.0, 1.0
    a = (1.0 - level) / 2.0
    lo = float(beta_dist.ppf(a,     k + 0.5, n - k + 0.5))
    hi = float(beta_dist.ppf(1 - a, k + 0.5, n - k + 0.5))
    return lo, hi

def build_gap_observations(tweet_times):
    """One row per inter-tweet gap with class label (see module notes, #1)
    and activity covariates (tweets in the W hours up to and including the
    gap-starting tweet, burst-inclusive)."""
    times = tweet_times.drop_duplicates().sort_values().reset_index(drop=True)
    rows = []
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        gap_min = (t1 - t0).total_seconds() / 60.0
        s0 = session_hour(t0)
        end_s = s0 + gap_min / 60.0
        anchor = night_anchor_date(t0)
        long_gap = gap_min >= SLEEP_GAP_HOURS * 60.0
        if long_gap and end_s >= NIGHT_END_SESSION_H:
            kind = "terminal"
        elif long_gap and EARLIEST_BEDTIME_H <= s0 <= LATEST_BEDTIME_H:
            kind = "shortnight"  # short night sleep that ended before 5 AM (#1)
        else:
            kind = "awake"
        rows.append({
            "t0": t0, "gap_min": gap_min, "s0": s0, "end_s": end_s,
            "hour_bin": int(s0), "anchor": anchor,
            "weekday": anchor.strftime("%A"),
            "kind": kind,
        })
    g = pd.DataFrame(rows)

    events = np.sort(tweet_times.values.astype("datetime64[ns]"))
    t0v = g["t0"].values.astype("datetime64[ns]")
    for w in ACTIVITY_WINDOWS_H:
        hi = np.searchsorted(events, t0v, side="right")
        lo = np.searchsorted(events, t0v - np.timedelta64(w * 60, "m"), side="right")
        g[f"act_{w}h"] = (hi - lo).astype(int)
    return g


NIGHT_KINDS = ("terminal", "shortnight")   # gaps that are sleeps "now" (#1)


class SleepStateModel:
    def __init__(self, gap_obs):
        self.gaps = gap_obs
        self.core = gap_obs                       # all kinds are observable (#1)
        self._reslice()
        self.act_sel = None
        self.sleep_bounds = {}
        self.dur_bounds = {}
        self._fold_of = self._contiguous_folds(self.core)
        try:
            self.kappa = self._tune_kappa()          # (#8)
        except Exception:
            self.kappa = float(SHRINK_KAPPA)
        # (#8) weekday log-odds main effects, per regime side and per outcome
        self._wd_offsets = {
            (side, outcome): self._weekday_offsets(side, outcome)
            for side in ("night", "day") for outcome in ("term", "night_gap")
        }

    def _reslice(self):
        self.awake = self.core[self.core["kind"] == "awake"]
        self.longnight = self.core[(self.core["kind"].isin(NIGHT_KINDS)) &
                                   (self.core["end_s"] <= MAX_WAKE_SESSION_H)]
        self.terminal = self.core[(self.core["kind"] == "terminal") &
                                  (self.core["end_s"] <= MAX_WAKE_SESSION_H)]

    # ── shrinkage / fold helpers ─────────────────────────────────────────────
    def _shrink(self, k, n, prior_p):
        return (k + self.kappa * prior_p) / (n + self.kappa)

    @staticmethod
    def _contiguous_folds(frame):
        """Contiguous time blocks (#7): adjacent nights share a fold, so
        multi-day regime autocorrelation cannot leak across train/test."""
        anchors = sorted(frame["anchor"].unique())
        n = max(len(anchors), 1)
        return {a: min(ACTIVITY_CV_FOLDS - 1, i * ACTIVITY_CV_FOLDS // n)
                for i, a in enumerate(anchors)}

    def _tune_kappa(self):
        """Pick kappa by grouped block-CV log-loss of the hour-conditional
        terminal model — the same shrinkage structure the live estimator
        uses (#8)."""
        g = self.core
        folds = g["anchor"].map(self._fold_of)
        eps = 1e-6
        best_k, best_ll = float(SHRINK_KAPPA), -np.inf
        for kappa in KAPPA_GRID:
            ll = 0.0
            for f in range(ACTIVITY_CV_FOLDS):
                tr, te = g[folds != f], g[folds == f]
                if tr.empty or te.empty:
                    continue
                ytr = (tr["kind"] == "terminal").astype(float)
                p_glob = ytr.mean()
                agg = tr.assign(_y=ytr).groupby("hour_bin")["_y"].agg(["sum", "count"])
                p_bin = (agg["sum"] + kappa * p_glob) / (agg["count"] + kappa)
                pb = te["hour_bin"].map(p_bin).fillna(p_glob).clip(eps, 1 - eps).values
                yte = (te["kind"] == "terminal").values.astype(float)
                ll += float((yte * np.log(pb) + (1 - yte) * np.log(1 - pb)).sum())
            if ll > best_ll:
                best_ll, best_k = ll, float(kappa)
        return best_k

    def _weekday_offsets(self, side, outcome):
        """Weekday log-odds main effects from ALL gaps on a regime side (#8):
        far more data per weekday than v1's evidence-cell slivers. outcome is
        'term' (until-morning event) or 'night_gap' (asleep-now event)."""
        if side == "night":
            sub = self.core[self.core["s0"] >= SLEEP_ZONE_START_S]
        else:
            sub = self.core[self.core["s0"] < SLEEP_ZONE_START_S]
        if outcome == "term":
            y = (sub["kind"] == "terminal")
        else:
            y = sub["kind"].isin(NIGHT_KINDS)
        p_all = float(y.mean()) if len(sub) else 0.5
        out = {}
        for d in DAYS_ORDER:
            m = (sub["weekday"] == d)
            k, n = int(y[m].sum()), int(m.sum())
            p_d = self._shrink(k, n, p_all)
            out[d] = _logit(p_d) - _logit(p_all)
        return out

    # ── evidence pools (#2) ──────────────────────────────────────────────────
    def _evidence_pool(self, last_s, silence_min, frame=None, tol=EVIDENCE_S0_TOL_H):
        """Gaps matching the evidence: started within ±tol (session hours) of
        the last tweet's clock time AND silent at least as long as observed.
        Returns rows and triangular kernel weights (floor 0.1) in start-time
        distance. This conditions on the observed silence directly, unlike
        v1's floor-bin / end-time proxy that mixed survival amounts."""
        fr = self.core if frame is None else frame
        d = (fr["s0"] - last_s).abs()
        m = (d <= tol) & (fr["gap_min"] >= silence_min)
        sub = fr[m]
        w = 1.0 - 0.9 * (d[m] / max(tol, 1e-9))
        return sub, w

    # ── Activity covariate machinery ────────────────────────────────────────
    @staticmethod
    def _tier_from_bounds(value, bounds):
        if bounds is None or value is None:
            return None
        q1, q2 = bounds
        return 0 if value <= q1 else (1 if value <= q2 else 2)

    @staticmethod
    def _bin_bounds(frame, col):
        out = {}
        for hb, grp in frame.groupby("hour_bin"):
            if len(grp) >= ACTIVITY_MIN_BIN_N:
                out[hb] = (float(grp[col].quantile(1 / 3)), float(grp[col].quantile(2 / 3)))
        return out

    @staticmethod
    def _assign_tiers(frame, col, bounds):
        q1 = frame["hour_bin"].map({hb: b[0] for hb, b in bounds.items()})
        q2 = frame["hour_bin"].map({hb: b[1] for hb, b in bounds.items()})
        v = frame[col]
        tier = pd.Series(np.where(v <= q1, 0.0, np.where(v <= q2, 1.0, 2.0)),
                         index=frame.index)
        tier[q1.isna()] = np.nan
        return tier

    def select_activity_windows(self):
        """Grouped CONTIGUOUS-BLOCK K-fold CV over candidate look-back windows
        (#7). Adoption rule per window: mean CV skill minus one fold-SE > 0
        AND >= ACTIVITY_MIN_FOLDS_POS folds positive (winner's-curse guard).
        Sleep-onset scored on all non-ambiguous gaps (Bernoulli log-loss);
        duration scored on AWAKE gaps only — the deployed quantity (#7)."""
        if self.act_sel is not None:
            return self.act_sel

        g = self.core.copy()
        g["fold"] = g["anchor"].map(self._fold_of)
        g["is_term"] = (g["kind"] == "terminal").astype(float)
        g["log_dur"] = np.log(g["gap_min"].clip(lower=0.5))
        eps = 1e-6

        table = []
        for w in ACTIVITY_WINDOWS_H:
            col = f"act_{w}h"
            ll_rates, mse_skills = [], []
            ll_gain_tot, n_ll = 0.0, 0
            sse_base_tot, sse_act_tot = 0.0, 0.0
            for fold in range(ACTIVITY_CV_FOLDS):
                tr = g[g["fold"] != fold]
                te = g[g["fold"] == fold]
                if tr.empty or te.empty:
                    continue
                bounds = self._bin_bounds(tr, col)
                tr = tr.assign(tier=self._assign_tiers(tr, col, bounds))
                te = te.assign(tier=self._assign_tiers(te, col, bounds))

                # — sleep-onset models —
                p_glob = tr["is_term"].mean()
                agg = tr.groupby("hour_bin")["is_term"].agg(["sum", "count"])
                p_bin = (agg["sum"] + self.kappa * p_glob) / (agg["count"] + self.kappa)
                tagg = (tr.dropna(subset=["tier"])
                          .groupby(["hour_bin", "tier"])["is_term"].agg(["sum", "count"]))
                tier_p = {hb_t: (r["sum"] + self.kappa * p_bin.get(hb_t[0], p_glob))
                                / (r["count"] + self.kappa)
                          for hb_t, r in tagg.iterrows()}
                pb = te["hour_bin"].map(p_bin).fillna(p_glob).values
                pt = np.array([tier_p.get((hb, ti), pb[i])
                               for i, (hb, ti) in enumerate(zip(te["hour_bin"], te["tier"]))])
                y = te["is_term"].values
                llb = y * np.log(np.clip(pb, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - pb, eps, 1 - eps))
                lla = y * np.log(np.clip(pt, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - pt, eps, 1 - eps))
                gain = lla.sum() - llb.sum()
                ll_gain_tot += gain
                n_ll += len(te)
                ll_rates.append(gain / max(len(te), 1))

                # — duration models: awake gaps only (#7) —
                tr_aw = tr[tr["kind"] == "awake"]
                te_aw = te[te["kind"] == "awake"]
                if tr_aw.empty or te_aw.empty:
                    continue
                m_glob = tr_aw["log_dur"].mean()
                dagg = tr_aw.groupby("hour_bin")["log_dur"].agg(["sum", "count"])
                m_bin = (dagg["sum"] + self.kappa * m_glob) / (dagg["count"] + self.kappa)
                dtagg = (tr_aw.dropna(subset=["tier"])
                           .groupby(["hour_bin", "tier"])["log_dur"].agg(["sum", "count"]))
                tier_m = {hb_t: (r["sum"] + self.kappa * m_bin.get(hb_t[0], m_glob))
                                / (r["count"] + self.kappa)
                          for hb_t, r in dtagg.iterrows()}
                mb = te_aw["hour_bin"].map(m_bin).fillna(m_glob).values
                mt = np.array([tier_m.get((hb, ti), mb[i])
                               for i, (hb, ti) in enumerate(zip(te_aw["hour_bin"], te_aw["tier"]))])
                yd = te_aw["log_dur"].values
                sb, sa = ((yd - mb) ** 2).sum(), ((yd - mt) ** 2).sum()
                sse_base_tot += sb
                sse_act_tot += sa
                mse_skills.append(1.0 - sa / max(sb, 1e-12))

            ll_rate = ll_gain_tot / max(n_ll, 1)
            ll_se = (np.std(ll_rates, ddof=1) / np.sqrt(len(ll_rates))
                     if len(ll_rates) > 1 else np.inf)
            mse_skill = 1.0 - sse_act_tot / max(sse_base_tot, 1e-12)
            mse_se = (np.std(mse_skills, ddof=1) / np.sqrt(len(mse_skills))
                      if len(mse_skills) > 1 else np.inf)
            ll_pos = sum(1 for r in ll_rates if r > 0)
            mse_pos = sum(1 for s in mse_skills if s > 0)
            table.append({
                "W": w,
                "ll_gain_per_gap": ll_rate, "ll_se": ll_se, "ll_folds_pos": ll_pos,
                "ll_adopt": (ll_rate - ll_se > 0) and (ll_pos >= ACTIVITY_MIN_FOLDS_POS),
                "mse_skill": mse_skill, "mse_se": mse_se, "mse_folds_pos": mse_pos,
                "mse_adopt": (mse_skill - mse_se > 0) and (mse_pos >= ACTIVITY_MIN_FOLDS_POS),
            })

        sel_df = pd.DataFrame(table)
        ll_ok = sel_df[sel_df["ll_adopt"]]
        ms_ok = sel_df[sel_df["mse_adopt"]]
        w_sleep = int(ll_ok.loc[ll_ok["ll_gain_per_gap"].idxmax(), "W"]) if len(ll_ok) else None
        w_dur = int(ms_ok.loc[ms_ok["mse_skill"].idxmax(), "W"]) if len(ms_ok) else None

        # Final full-data tier boundaries / assignments for live use.
        if w_sleep is not None:
            self.sleep_bounds = self._bin_bounds(self.gaps, f"act_{w_sleep}h")
            self.gaps["tier_sleep"] = self._assign_tiers(self.gaps, f"act_{w_sleep}h", self.sleep_bounds)
        else:
            self.gaps["tier_sleep"] = np.nan
        if w_dur is not None:
            self.dur_bounds = self._bin_bounds(self.gaps, f"act_{w_dur}h")
            self.gaps["tier_dur"] = self._assign_tiers(self.gaps, f"act_{w_dur}h", self.dur_bounds)
        else:
            self.gaps["tier_dur"] = np.nan
        self.core = self.gaps
        self._reslice()

        night = self.core[(self.core["hour_bin"] >= EARLIEST_BEDTIME_H) &
                          (self.core["hour_bin"] <= LATEST_BEDTIME_H)]
        effects = {}
        if w_sleep is not None:
            sub = night.dropna(subset=["tier_sleep"])
            effects["sleep"] = {
                int(t): {"p_term": float((grp["kind"] == "terminal").mean()), "n": len(grp)}
                for t, grp in sub.groupby("tier_sleep")
            }
        if w_dur is not None:
            sub = night[night["kind"] == "awake"].dropna(subset=["tier_dur"])
            effects["dur"] = {
                int(t): {"median_gap": float(grp["gap_min"].median()), "n": len(grp)}
                for t, grp in sub.groupby("tier_dur")
            }

        self.act_sel = {"table": sel_df, "w_sleep": w_sleep, "w_dur": w_dur,
                        "effects": effects}
        return self.act_sel

    # ── posterior (#1, #2, #4, #8) ───────────────────────────────────────────
    def p_asleep(self, weekday, last_s, silence_min, activity_count=None):
        """Two posteriors from one kernel evidence pool (#1):
          primary return — P(asleep: currently in a night-rest gap), i.e.
                           positive = terminal OR shortnight;
          components['p_term'] (+band) — P(no more tweets until 5 AM), i.e.
                           positive = terminal only (the betting event).
        Pool conditioned on the observed silence (#2), kappa-shrunk through
        regime base → pool → activity tier; weekday applied as a log-odds
        main effect (#8); credible band = the Beta posterior implied by the
        shrinkage prior, shifted by the same weekday offset (#4)."""
        silence_min = max(silence_min, 0.0)
        tau = last_s + silence_min / 60.0
        night_side = last_s >= SLEEP_ZONE_START_S
        side_key = "night" if night_side else "day"

        side = (self.core[self.core["s0"] >= SLEEP_ZONE_START_S] if night_side
                else self.core[self.core["s0"] < SLEEP_ZONE_START_S])
        base = side[side["gap_min"] >= silence_min]

        pool, w = self._evidence_pool(last_s, silence_min)
        wv = w.values

        # Activity-tier mask (only when a CV-validated window exists). Tier
        # labels are within-bin terciles, hence comparable across the bins a
        # kernel window may span.
        tier, tier_mask = None, None
        if (activity_count is not None and self.act_sel is not None
                and self.act_sel["w_sleep"] is not None):
            tier = self._tier_from_bounds(activity_count, self.sleep_bounds.get(int(last_s)))
            if tier is not None and "tier_sleep" in pool.columns:
                tier_mask = (pool["tier_sleep"] == float(tier)).values

        def estimate(pos_pool, pos_base, outcome):
            p_base = float(pos_base.mean()) if len(base) else 0.5
            k1, n1 = float((wv * pos_pool).sum()), float(wv.sum())
            p_pool = self._shrink(k1, n1, p_base)
            ctx_k, ctx_n, p_ctx, prior_ctx = k1, n1, p_pool, p_base
            if tier_mask is not None:
                kt = float((wv * pos_pool * tier_mask).sum())
                nt = float((wv * tier_mask).sum())
                p_ctx = self._shrink(kt, nt, p_pool)
                ctx_k, ctx_n, prior_ctx = kt, nt, p_pool
            delta = self._wd_offsets[(side_key, outcome)].get(weekday, 0.0)
            p = _sigmoid(_logit(p_ctx) + delta)
            alpha = ctx_k + self.kappa * prior_ctx
            beta_ = (ctx_n - ctx_k) + self.kappa * (1.0 - prior_ctx)
            lo = _sigmoid(_logit(float(beta_dist.ppf(0.025, alpha, beta_))) + delta)
            hi = _sigmoid(_logit(float(beta_dist.ppf(0.975, alpha, beta_))) + delta)
            return (p, min(lo, p), max(hi, p),
                    {"k": round(ctx_k, 1), "n": round(ctx_n, 1),
                     "k_pool": round(k1, 1), "n_pool": round(n1, 1),
                     "p_pool": p_pool, "p_base": p_base, "delta": delta})

        pos_night = pool["kind"].isin(NIGHT_KINDS).values.astype(float)
        pos_term = (pool["kind"] == "terminal").values.astype(float)
        p_n, lo_n, hi_n, d_n = estimate(pos_night, base["kind"].isin(NIGHT_KINDS), "night_gap")
        p_t, lo_t, hi_t, d_t = estimate(pos_term, (base["kind"] == "terminal"), "term")

        components = {"k_wd": d_n["k"], "n_wd": d_n["n"],
                      "k_pool": d_n["k_pool"], "n_pool": d_n["n_pool"],
                      "p_pool": d_n["p_pool"], "p_base": d_n["p_base"],
                      "n_base": len(base), "tau": tau, "tier": tier,
                      "p_tier": d_n["p_pool"], "n_tier": d_n["n"],
                      "delta_wd": d_n["delta"],
                      "p_term": p_t, "term_lo": lo_t, "term_hi": hi_t,
                      "term_p_base": d_t["p_base"], "term_delta": d_t["delta"],
                      "term_k_pool": d_t["k_pool"]}
        return p_n, lo_n, hi_n, components

    def _marginal_occupancy(self, cur_s, kinds):
        """Marginal (clock-hour) occupancy probability, NOT conditioned on the
        last tweet: the share of historical nights in which a gap of one of
        `kinds` spanned session-hour cur_s. Used as the live fallback when the
        silence is too long to condition on (e.g. a multi-day silence that has
        wrapped past the daytime into a new night). Returns (p, lo, hi) with a
        Jeffreys band over the nights observed."""
        sub = self.core[self.core["kind"].isin(kinds)]
        nights = max(self.core["anchor"].nunique(), 1)
        cover = sub[(sub["s0"] <= cur_s) & (sub["end_s"] >= cur_s)]
        k = int(cover["anchor"].nunique())
        lo, hi = jeffreys_interval(k, nights)
        return k / nights, lo, hi

    def marginal_p_asleep(self, cur_s):
        """Marginal P(in a night-rest gap | clock hour) — terminal OR
        shortnight. ~0% in the daytime, realistic at night."""
        return self._marginal_occupancy(cur_s, NIGHT_KINDS)

    def marginal_p_terminal(self, cur_s):
        """Marginal P(no more tweets until 5 AM | clock hour): the share of
        nights in which a TERMINAL gap (one that lasted until >= 5 AM) spanned
        this clock hour. A shortnight gap ends before 5 AM, so it is correctly
        excluded here; hence this is always <= marginal_p_asleep."""
        return self._marginal_occupancy(cur_s, ("terminal",))

    # ── predictive distribution of the next tweet (#5) ──────────────────────
    def _predictive_pool(self, weekday, last_s, silence_min, act_dur=None):
        """(vals, wts, tau) for the next-tweet predictive in the LAST-TWEET
        session frame: end times of evidence-matched gaps with weights =
        triangular start-time kernel × WEEKDAY_POOL_WEIGHT (same weekday) ×
        TIER_POOL_WEIGHT (matching activity tier). The kernel widens until the
        pool is adequately populated; final fallback is the bedtime-agnostic
        wake distribution. Shared by the quantile and Monte Carlo predictors."""
        silence_min = max(silence_min, 0.0)
        tau = last_s + silence_min / 60.0
        tier_d = None
        if (act_dur is not None and self.act_sel is not None
                and self.act_sel["w_dur"] is not None):
            tier_d = self._tier_from_bounds(act_dur, self.dur_bounds.get(int(last_s)))

        pool, w = None, None
        for tol in EVIDENCE_TOL_WIDEN:
            pool, w = self._evidence_pool(last_s, silence_min, tol=tol)
            if len(pool) >= 8:
                break
        if pool is None or len(pool) == 0:
            wk = self.longnight[self.longnight["end_s"] > tau]
            if len(wk) == 0:
                return np.array([]), np.array([]), tau
            return wk["end_s"].values.astype(float), np.ones(len(wk)), tau

        vals = pool["end_s"].values.astype(float)
        wts = w.values.astype(float).copy()
        wts = wts * np.where(pool["weekday"].values == weekday, WEEKDAY_POOL_WEIGHT, 1.0)
        if tier_d is not None and "tier_dur" in pool.columns:
            wts = wts * np.where(pool["tier_dur"].values == float(tier_d),
                                 TIER_POOL_WEIGHT, 1.0)
        return vals, wts, tau

    def _daytime_predictive_pool(self, last_s, silence_min, cur_s,
                                 last_anchor, now_anchor):
        """Daytime (out-of-night) next-tweet predictor. Conditions on the
        elapsed silence where that helps (backtest: ~18% lower mean error on
        the cases it changes, no worse elsewhere); otherwise stays memoryless.

        Preferred — SILENCE-CONDITIONED: awake gaps that start near the last
        tweet's hour AND already lasted at least the elapsed silence. The next
        tweet is read off their end times, so short gaps he has already
        outlasted are ruled out (residual-life conditioning). Anchored at the
        last tweet's night-date.

        Fallback — MEMORYLESS: when the silence is so long that too few awake
        gaps match it (e.g. a night-spanning silence), draw a fresh awake gap
        near the CURRENT clock hour instead. Anchored at 'now'.

        Returns (vals, wts, tau, anchor)."""
        silence_min = max(silence_min, 0.0)
        aw = self.awake

        # silence-conditioned attempt (last-tweet session frame)
        for tol in EVIDENCE_TOL_WIDEN:
            d = (aw["s0"] - last_s).abs()
            m = (d <= tol) & (aw["gap_min"] >= silence_min)
            if int(m.sum()) >= MC_DAYTIME_MIN_POOL:
                break
        if int(m.sum()) >= MC_DAYTIME_MIN_POOL:
            sub, dd = aw[m], d[m]
            vals = sub["end_s"].values.astype(float)
            wts = (1.0 - 0.9 * (dd / max(tol, 1e-9))).values.astype(float)
            return vals, wts, last_s + silence_min / 60.0, last_anchor

        # memoryless fallback (current-clock session frame)
        for tol in EVIDENCE_TOL_WIDEN:
            d = (aw["s0"] - cur_s).abs()
            m = d <= tol
            if int(m.sum()) >= MC_DAYTIME_MIN_POOL:
                break
        sub, dd = aw[m], d[m]
        if len(sub) == 0:
            return np.array([]), np.array([]), cur_s, now_anchor
        vals = cur_s + sub["gap_min"].values.astype(float) / 60.0
        wts = (1.0 - 0.9 * (dd / max(tol, 1e-9))).values.astype(float)
        return vals, wts, cur_s, now_anchor

    def next_tweet_quantiles(self, weekday, last_s, silence_min, act_dur=None):
        """Weighted quantiles of the next-tweet session-hour (see
        _predictive_pool)."""
        vals, wts, tau = self._predictive_pool(weekday, last_s, silence_min, act_dur)
        if len(vals) == 0:
            return {q: tau + 0.5 for q in PRED_QUANTILES}, 0.0
        qs = _weighted_quantiles(vals, wts, PRED_QUANTILES)
        return dict(zip(PRED_QUANTILES, qs)), float(np.sum(wts))

    # ── Monte Carlo next-tweet predictive (v3) ──────────────────────────────
    def next_tweet_montecarlo(self, vals, wts, tau, target_s=None,
                              windows_min=None, n_samples=MC_N_SAMPLES,
                              n_boot=MC_BOOTSTRAP, jitter_min=MC_JITTER_MIN,
                              bucket_max_h=MC_BUCKET_MAX_H, seed=MC_SEED,
                              epistemic=True):
        """Monte Carlo predictive for the next-tweet session-hour.

        vals : candidate next-tweet session-hours (gap end times).
        wts  : non-negative pool weights (the empirical mixture).
        tau  : current time in session hours; the next tweet must be > tau.

        Aleatoric layer — draw n_samples tweet times from the weighted pool,
        each perturbed by N(0, jitter_min) timing noise and reflected across
        tau so every draw lies in the future; these give the predictive
        quantiles, the hour-by-hour profile and the point window-probabilities.
        Epistemic layer (epistemic=True only) — n_boot weighted bootstrap
        resamples of the finite HISTORICAL pool give a credible band on each
        window-probability. Pass epistemic=False when `vals` are model-generated
        predictive draws (e.g. the EVT tail): bootstrapping them would only
        re-measure Monte-Carlo noise and report a spuriously tight band, so the
        band is omitted (its uncertainty already lives in the sample spread).

        Returns a results dict (session-hour units), or None if the pool is
        empty.
        """
        if windows_min is None:
            windows_min = NEXT_TWEET_WINDOWS_MIN
        vals = np.asarray(vals, dtype=float)
        wts = np.maximum(np.asarray(wts, dtype=float), 0.0)
        if len(vals) == 0 or wts.sum() <= 0:
            return None
        rng = np.random.default_rng(seed)
        p = wts / wts.sum()
        sigma_h = max(jitter_min, 0.0) / 60.0

        # ── aleatoric predictive samples ──
        idx = rng.choice(len(vals), size=n_samples, p=p)
        samp = vals[idx] + (rng.normal(0.0, sigma_h, size=n_samples)
                            if sigma_h > 0 else 0.0)
        below = samp < tau
        samp[below] = 2.0 * tau - samp[below]      # reflect into the future
        np.maximum(samp, tau, out=samp)            # safety floor

        qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        quant = {q: float(np.quantile(samp, q)) for q in qs}
        mean_s = float(samp.mean())
        if target_s is None:
            target_s = quant[0.50]
        target_s = float(target_s)

        # ── window probabilities ──
        # The timing jitter is integrated ANALYTICALLY: for a pool time v, a
        # jittered draw lands within ±w of the target with probability
        # Φ((target+w−v)/σ) − Φ((target−w−v)/σ). Averaging that over the pool
        # (weighted) gives the point estimate; averaging over each bootstrap
        # resample gives the credible band. Doing both the same way keeps the
        # point estimate inside its own band (a plain jittered-vs-raw mix did
        # not). With σ→0 it degrades to the raw indicator.
        def _win_g(arr, w_h):
            if sigma_h <= 0:
                return (np.abs(arr - target_s) <= w_h).astype(float)
            return (norm_dist.cdf((target_s + w_h - arr) / sigma_h)
                    - norm_dist.cdf((target_s - w_h - arr) / sigma_h))

        window_probs = {}
        if epistemic:
            boot = rng.choice(len(vals), size=(n_boot, len(vals)), p=p)
            boot_vals = vals[boot]                 # (n_boot, n_pool)
            for wmin in windows_min:
                w_h = wmin / 60.0
                pt = float(np.dot(p, _win_g(vals, w_h)))
                bp = _win_g(boot_vals.reshape(-1), w_h).reshape(boot_vals.shape).mean(axis=1)
                window_probs[int(wmin)] = (pt, float(np.quantile(bp, 0.05)),
                                           float(np.quantile(bp, 0.95)))
        else:
            # Model-generated draws: report the point only (band would be MC noise).
            for wmin in windows_min:
                pt = float(np.dot(p, _win_g(vals, wmin / 60.0)))
                window_probs[int(wmin)] = (pt, None, None)

        p_before = float(np.mean(samp < target_s))

        # ── hour-by-hour probability profile from tau forward ──
        h0 = int(np.floor(tau))
        buckets, cum, h = [], 0.0, h0
        while h < h0 + bucket_max_h and cum < 0.995:
            pmass = float(np.mean((samp >= h) & (samp < h + 1)))
            buckets.append((float(h), pmass))
            cum += pmass
            h += 1
        peak = max(buckets, key=lambda b: b[1]) if buckets else (tau, 0.0)

        return {
            "n_pool": int(len(vals)), "pool_weight": float(wts.sum()),
            "quantiles": quant, "mean_s": mean_s, "target_s": target_s,
            "window_probs": window_probs, "p_before_target": p_before,
            "hour_buckets": buckets, "peak_bucket": peak,
            "mc_n": int(n_samples), "boot_n": int(n_boot),
            "jitter_min": float(jitter_min),
        }

    # ── two-branch expectation (secondary diagnostic; #3 wake conditioning) ──
    def expected_next_tweet(self, weekday, last_s, silence_min,
                            act_sleep=None, act_dur=None):
        """Posterior-weighted expected session-hour of the next tweet. The
        sleep branch (terminal + shortnight ends) conditions on bedtime
        proximity (±WAKE_BEDTIME_TOL_H, widening; #3); the awake branch uses
        kernel-matched awake gaps."""
        p_sleep, _, _, _ = self.p_asleep(weekday, last_s, silence_min, act_sleep)
        silence_min = max(silence_min, 0.0)
        tau = last_s + silence_min / 60.0

        bd = (self.longnight["s0"] - last_s).abs()
        wake_cands = [
            self.longnight[(self.longnight["weekday"] == weekday) & (bd <= WAKE_BEDTIME_TOL_H)],
            self.longnight[bd <= WAKE_BEDTIME_TOL_H],
            self.longnight[(self.longnight["weekday"] == weekday) & (bd <= 2 * WAKE_BEDTIME_TOL_H)],
            self.longnight[bd <= 2 * WAKE_BEDTIME_TOL_H],
            self.longnight[self.longnight["weekday"] == weekday],
            self.longnight,
        ]
        wake_exp = None
        for c in wake_cands:
            v = c.loc[c["end_s"] > tau, "end_s"]
            if len(v) >= 3:
                wake_exp = float(v.mean())
                break
        if wake_exp is None:
            wake_exp = max(tau + 0.5, 33.0)

        tier_d = None
        if (act_dur is not None and self.act_sel is not None
                and self.act_sel["w_dur"] is not None):
            tier_d = self._tier_from_bounds(act_dur, self.dur_bounds.get(int(last_s)))
        pool, w = None, None
        for tol in EVIDENCE_TOL_WIDEN:
            pool, w = self._evidence_pool(last_s, silence_min, frame=self.awake, tol=tol)
            if tier_d is not None and "tier_dur" in pool.columns:
                tp = pool[pool["tier_dur"] == float(tier_d)]
                if len(tp) >= 5:
                    pool, w = tp, w.loc[tp.index]
            if len(pool) >= 5:
                break
        if pool is not None and len(pool):
            residual_min = float(np.average(pool["gap_min"].values,
                                            weights=np.maximum(w.values, 1e-6))) - silence_min
            residual_min = max(residual_min, 1.0)
        else:
            residual_min = 15.0
        awake_exp = tau + residual_min / 60.0

        combined = p_sleep * wake_exp + (1.0 - p_sleep) * awake_exp
        return combined, awake_exp, wake_exp, p_sleep


class SilenceTailModel:
    """Extreme-value model for an ongoing silence that has run past the record
    (v4). Deseasonalizes inter-tweet gaps by the circadian intensity μ(hour)
    via the time-rescaling theorem, fits a Peaks-Over-Threshold Generalized
    Pareto tail (Bayesian grid posterior, weekday partial pooling) to the
    rescaled gaps, and uses GPD threshold-stability for the residual-life law,
    mapped back to wall-clock through the inverse intensity. A competing-risk
    'dormant' mixture widens the predictive past the historical maximum."""

    def __init__(self, tweet_times):
        times = tweet_times.drop_duplicates().sort_values().reset_index(drop=True)
        self.times = times
        self._build_intensity(times)
        self._build_rescaled_gaps(times)
        self._posterior_cache = {}

    # ── circadian intensity μ(hour) and (inverse) cumulative intensity ────────
    def _build_intensity(self, times):
        """μ(h) = mean tweets per clock-hour h, in tweets/hour. Floored above 0
        so the dead hours still carry a trickle (keeps the inverse well-posed)."""
        if len(times) == 0:
            self.mu = np.full(24, 1.0); self.mu_total = 24.0; return
        hours = times.dt.hour.values
        n_days = max(len(set(times.dt.date)), 1)
        counts = np.array([(hours == h).sum() for h in range(24)], dtype=float)
        mu = counts / n_days
        self.mu = np.maximum(mu, 1e-3)
        self.mu_total = float(self.mu.sum())          # expected tweets/day

    @staticmethod
    def _next_hour(cur):
        """Start of the next EST clock hour. Uses .replace (field edit, no
        re-localization) so it is tolerant of the ambiguous/nonexistent DST
        hours — .floor()/.ceil() would raise on the fall-back hour."""
        return cur.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    def _cum_intensity(self, t0, t1):
        """∫ μ dt over [t0, t1] in 'expected tweets' units (0 if t1<=t0).

        Steps on EST clock-hour boundaries; the EST wall-clock hour drives μ.
        The inverse (_invert_grid) uses the SAME stepping, so the rescale↔
        wall-clock round trip is exact. (At the two DST transitions per year the
        hour-binned integral carries a benign ≤1 h approximation.)"""
        t0, t1 = pd.Timestamp(t0), pd.Timestamp(t1)
        if t1 <= t0:
            return 0.0
        total, cur = 0.0, t0
        while cur < t1:
            seg_end = min(self._next_hour(cur), t1)
            total += self.mu[cur.hour] * ((seg_end - cur).total_seconds() / 3600.0)
            cur = seg_end
        return total

    def _build_rescaled_gaps(self, times):
        rg, wd = [], []
        for i in range(len(times) - 1):
            t0, t1 = times.iloc[i], times.iloc[i + 1]
            rg.append(self._cum_intensity(t0, t1))
            wd.append(night_anchor_date(t0).strftime("%A"))
        self.rgaps = np.asarray(rg, dtype=float)
        self.rgap_weekday = np.asarray(wd)
        self.rgap_max = float(self.rgaps.max()) if len(rg) else 0.0
        self.u = float(np.quantile(self.rgaps, TAIL_THRESHOLD_Q)) if len(rg) else 0.0
        self.n_exceed = int((self.rgaps > self.u).sum())

    # ── Bayesian (grid) GPD posterior over the rescaled exceedances ──────────
    def _fit_posterior(self, target_weekday=None):
        if not self.usable():
            return None
        key = target_weekday or "_global"
        if key in self._posterior_cache:
            return self._posterior_cache[key]

        mask = self.rgaps > self.u
        y = self.rgaps[mask] - self.u
        w = np.ones(len(y))
        if target_weekday is not None:                # weekday partial pooling
            w = np.where(self.rgap_weekday[mask] == target_weekday,
                         WEEKDAY_TAIL_WEIGHT, 1.0)

        ybar = max(float(np.average(y, weights=w)), 1e-6)
        xi_grid = np.linspace(*TAIL_XI_GRID[:2], int(TAIL_XI_GRID[2]))
        sig_grid = np.linspace(TAIL_SIGMA_GRID_MULT[0] * ybar,
                               TAIL_SIGMA_GRID_MULT[1] * ybar, int(TAIL_SIGMA_GRID_MULT[2]))
        logpost = np.full((len(xi_grid), len(sig_grid)), -np.inf)
        for a, xi in enumerate(xi_grid):
            for b, sig in enumerate(sig_grid):
                ll = genpareto.logpdf(y, c=xi, loc=0.0, scale=sig)
                if np.all(np.isfinite(ll)):           # weak uniform prior on the grid
                    logpost[a, b] = float(np.sum(w * ll))
        if not np.isfinite(np.max(logpost)):          # no feasible (ξ,σ) on the grid
            self._posterior_cache[key] = None
            return None
        post = np.exp(logpost - np.max(logpost))
        post /= post.sum()

        rng = np.random.default_rng(MC_SEED)
        flat = post.ravel()
        idx = rng.choice(flat.size, size=TAIL_POSTERIOR_DRAWS, p=flat)
        ai, bi = np.unravel_index(idx, post.shape)
        xi_draws, sig_draws = xi_grid[ai], sig_grid[bi]
        summary = {
            "xi_med": float(np.median(xi_draws)),
            "xi_lo": float(np.quantile(xi_draws, 0.05)),
            "xi_hi": float(np.quantile(xi_draws, 0.95)),
            "sig_med": float(np.median(sig_draws)),
            "n_exceed": int(len(y)),
        }
        self._posterior_cache[key] = (xi_draws, sig_draws, summary)
        return self._posterior_cache[key]

    def usable(self):
        return self.n_exceed >= TAIL_MIN_EXCEEDANCES

    def rescaled_elapsed(self, now, last_tweet):
        """Deseasonalized elapsed silence (∫ μ from last tweet to now), in
        'expected tweets missed' units."""
        return self._cum_intensity(last_tweet, now)

    # ── residual-life prediction in the tail ─────────────────────────────────
    def _invert_grid(self, now):
        """Exact lookup (cumulative intensity, minutes-from-now) for mapping a
        rescaled residual back to wall-clock. Breakpoints sit on EST clock-hour
        boundaries, so np.interp reproduces the piecewise-linear inverse of
        _cum_intensity exactly (no discretization mismatch) and is DST-aware."""
        now = pd.Timestamp(now)
        end = now + timedelta(hours=TAIL_INVERT_HORIZON_H)
        cum_list, off_list = [0.0], [0.0]
        total, cur = 0.0, now
        while cur < end:
            nxt = min(self._next_hour(cur), end)
            total += self.mu[cur.hour] * ((nxt - cur).total_seconds() / 3600.0)
            off_list.append((nxt - now).total_seconds() / 60.0)
            cum_list.append(total)
            cur = nxt
        return np.asarray(cum_list), np.asarray(off_list)

    def predict(self, now, last_tweet, anchor, target_weekday,
                n_samples=TAIL_SAMPLES):
        """Posterior-predictive next-tweet session-hours (frame anchored at
        `anchor`) for an in-tail silence, plus diagnostics. Returns None if the
        tail model is unusable."""
        post = self._fit_posterior(target_weekday)
        if post is None:
            return None
        xi_d, sig_d, summary = post
        now = pd.Timestamp(now); last_tweet = pd.Timestamp(last_tweet)
        E = self._cum_intensity(last_tweet, now)             # rescaled elapsed silence
        rng = np.random.default_rng(MC_SEED + 7)

        di = rng.integers(0, len(xi_d), size=n_samples)
        xis, sigs = xi_d[di], sig_d[di]
        # threshold stability: residual above level E (>u) is GPD(σ+ξ(E-u), ξ)
        sig_p = np.maximum(sigs + xis * (E - self.u), 1e-9)
        U = rng.random(n_samples)
        small = np.abs(xis) < 1e-6
        delta = np.where(small, -sig_p * np.log1p(-U),
                         sig_p / np.where(small, 1.0, xis) * ((1.0 - U) ** (-xis) - 1.0))
        delta = np.maximum(delta, 0.0)                       # rescaled additional intensity

        # competing-risk "dormant" mixture once past the historical maximum
        regime_change = E > self.rgap_max
        p_dormant = 0.0
        if regime_change:
            frac = (E - self.rgap_max) / max(self.rgap_max - self.u, 1e-6)
            p_dormant = float(min(REGIME_CHANGE_PROB_MAX, 0.10 + 0.25 * frac))
            dormant = rng.random(n_samples) < p_dormant
            delta = np.where(dormant, delta + self.mu_total * rng.uniform(0.5, 2.0, n_samples),
                             delta)

        # map rescaled residual → wall-clock minutes from now → session hours
        cum, offs = self._invert_grid(now)
        off_min = np.interp(delta, cum, offs)               # clamps at horizon
        anchor_mid = EST.localize(datetime.combine(anchor, datetime.min.time()))
        base_h = (now - anchor_mid).total_seconds() / 3600.0
        samples_s = base_h + off_min / 60.0

        # P(no more tweets until 5 AM): analytic GPD survival per posterior draw
        # (carries a credible band), blended with the dormant competing risk.
        five = EST.localize(datetime.combine(now.date(), datetime.min.time())
                            + timedelta(hours=NIGHT_END_SESSION_H % 24))
        if five <= now:
            five += timedelta(days=1)
        delta_t = self._cum_intensity(now, five)             # rescaled intensity now→5 AM
        sig_t = np.maximum(sig_d + xi_d * (E - self.u), 1e-9)
        near0 = np.abs(xi_d) < 1e-6
        base = 1.0 + xi_d * delta_t / sig_t
        with np.errstate(invalid="ignore", divide="ignore"):
            s_gpd = np.where(base > 0, base ** (-1.0 / np.where(near0, 1.0, xi_d)), 0.0)
        surv = np.where(near0, np.exp(-delta_t / sig_t), s_gpd)
        # Dormant draws add mu_total·U(0.5,2) of rescaled intensity; they survive
        # to 5 AM only if that exceeds delta_t (matters when 5 AM is far away,
        # e.g. an already-morning anomaly), so weight the dormant term by that
        # probability instead of assuming it always reaches 5 AM.
        s_dormant = float(np.clip((2.0 - delta_t / max(self.mu_total, 1e-9)) / 1.5, 0.0, 1.0))
        p_until_draw = ((1.0 - p_dormant) * np.clip(surv, 0.0, 1.0)
                        + p_dormant * s_dormant)
        p_until = float(np.mean(p_until_draw))
        p_until_lo = float(np.quantile(p_until_draw, 0.05))
        p_until_hi = float(np.quantile(p_until_draw, 0.95))

        # median additional silence in wall-clock hours (for display)
        resid_off = np.interp(np.median(delta), cum, offs) / 60.0

        return {
            "samples_s": samples_s, "tau": base_h, "anchor": anchor,
            "E": E, "u": self.u, "rgap_max": self.rgap_max,
            "xi_med": summary["xi_med"], "xi_lo": summary["xi_lo"],
            "xi_hi": summary["xi_hi"], "n_exceed": summary["n_exceed"],
            "regime_change": bool(regime_change), "p_dormant": p_dormant,
            "p_until_am": p_until, "p_until_lo": p_until_lo, "p_until_hi": p_until_hi,
            "resid_median_h": float(resid_off),
            "E_pct": float((self.rgaps <= E).mean()),
        }


def direct_empirical_check(tweet_times, weekday, last_s, silence_min):
    """Model-free cross-check: among historical nights of this weekday where
    the last tweet fell within ±45 min of the same clock time and silence had
    reached the query time, how many stayed silent until the morning
    (next tweet at/after NIGHT_END_SESSION_H, i.e. 5 AM)?"""
    times = tweet_times.drop_duplicates().sort_values().reset_index(drop=True)
    tau = last_s + silence_min / 60.0
    anchors = sorted({night_anchor_date(t) for t in times})
    matches, asleep = 0, 0
    for a in anchors:
        if a.strftime("%A") != weekday:
            continue
        base_naive = datetime.combine(a, datetime.min.time()) + timedelta(hours=tau)
        try:
            base = pd.Timestamp(EST.localize(base_naive))
        except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError):
            continue
        idx = int(times.searchsorted(base)) - 1
        if idx < 0 or idx + 1 >= len(times):
            continue
        t_last = times.iloc[idx]
        if night_anchor_date(t_last) != a or abs(session_hour(t_last) - last_s) > DIRECT_MATCH_TOL_H:
            continue
        matches += 1
        nxt = times.iloc[idx + 1]
        still_same_night = (night_anchor_date(nxt) == a) and (session_hour(nxt) < NIGHT_END_SESSION_H)
        if not still_same_night:
            asleep += 1
    return asleep, matches


def session_to_datetime(anchor_date, s):
    """Convert a session-hour (hours since midnight of the anchor date) to an
    absolute EST datetime: 26.5 on anchor 2026-06-11 → 2026-06-12 02:30 EST."""
    naive = datetime.combine(anchor_date, datetime.min.time()) + timedelta(hours=s)
    return EST.localize(naive)


def count_recent_tweets(tweet_times, t_end, hours):
    """Tweets (burst-inclusive event counts) in (t_end - hours, t_end]."""
    arr = np.sort(tweet_times.values.astype("datetime64[ns]"))
    t1 = pd.Timestamp(t_end).to_datetime64()
    hi = np.searchsorted(arr, t1, side="right")
    lo = np.searchsorted(arr, t1 - np.timedelta64(int(hours * 60), "m"), side="right")
    return int(hi - lo)


def _resolve_target_s(anchor, target_cfg):
    """Translate NEXT_TWEET_TARGET into a session-hour in the prediction frame
    (hours since the anchor's midnight), or None to use the predicted median.
    Accepts a full 'M/D/YYYY HH:MM' datetime or a bare 'HH:MM' clock time."""
    if not target_cfg:
        return None
    d, h, m = parse_hhmm_or_datetime(target_cfg)
    base = d if d is not None else anchor
    naive = datetime.combine(base, datetime.min.time()) + timedelta(hours=h, minutes=m)
    return (naive - datetime.combine(anchor, datetime.min.time())).total_seconds() / 3600.0


def evaluate_current_state(model, tweet_times, now=None, last_tweet=None):
    """Single source of truth for the live estimate. v2 (#6): no hard daytime
    pin — p_terminal (P(no more tweets until tomorrow morning)) is reported in
    both regimes; the estimator's own bedtime prior keeps it near zero in the
    daytime without a discontinuity at the 10 PM boundary. The regime label is
    kept for display and the regime-matched base rate. Live silence is
    corrected by INGEST_LAG_MIN for CSV ingestion latency."""
    if now is None:
        now = datetime.now(EST)
    if last_tweet is None:
        last_tweet = tweet_times.iloc[-1]

    silence_raw = (pd.Timestamp(now) - pd.Timestamp(last_tweet)).total_seconds() / 60.0
    silence_min = max(0.0, silence_raw - INGEST_LAG_MIN)
    last_s = session_hour(last_tweet)
    anchor = night_anchor_date(last_tweet)
    weekday = anchor.strftime("%A")
    cur_s = session_hour(now)
    in_zone = SLEEP_ZONE_START_S <= cur_s <= SLEEP_ZONE_END_S

    sel = model.select_activity_windows()
    act_sleep = (count_recent_tweets(tweet_times, last_tweet, sel["w_sleep"])
                 if sel["w_sleep"] is not None else None)
    act_dur = (count_recent_tweets(tweet_times, last_tweet, sel["w_dur"])
               if sel["w_dur"] is not None else None)

    p_sleep, lo, hi, comp = model.p_asleep(weekday, last_s, silence_min, act_sleep)
    combined_s, awake_s, wake_s, _ = model.expected_next_tweet(
        weekday, last_s, silence_min, act_sleep, act_dur)
    q_s, q_neff = model.next_tweet_quantiles(weekday, last_s, silence_min, act_dur)

    tier_dur = (model._tier_from_bounds(act_dur, model.dur_bounds.get(int(last_s)))
                if act_dur is not None else None)

    # Cache the EVT tail model on the SleepStateModel (rebuilt only when the
    # model itself is rebuilt, i.e. when new tweets arrive).
    tail = getattr(model, "_tail_model", None)
    if tail is None:
        tail = SilenceTailModel(tweet_times)
        model._tail_model = tail

    # ── Regime selection for the live readout ───────────────────────────────
    #  tail     : the silence is BOTH deseasonalized-unusual (rescaled elapsed
    #             past the POT threshold) AND no longer conditionable — it has
    #             run past the wake window OR the direct evidence pool is nearly
    #             empty. Only then does the EVT residual-life model preempt the
    #             night-rest inference, so a routine peak-hour quiet (top-5%
    #             rescaled ≈ 2 h at 1 AM) stays in_night.
    #  daytime  : silence ran past the wake window but isn't extreme — marginal
    #             P(asleep)/P(until 5 AM) + a fresh-daytime-gap predictor.
    #  in_night : normal night-rest conditioning (#v2/#v3).
    out_of_night = (last_s + silence_min / 60.0) > NIGHT_OVER_SESSION_H
    E_resc = tail.rescaled_elapsed(now, last_tweet)
    cond_sub, _ = model._evidence_pool(last_s, silence_min)
    silence_is_tail = (tail.usable() and E_resc > tail.u
                       and (out_of_night or len(cond_sub) < TAIL_MIN_CONDITION_POOL))
    tail_info = None

    if silence_is_tail:
        mode = "tail"
        p_sleep, lo, hi = model.marginal_p_asleep(cur_s)
        comp["delta_wd"] = 0.0
        weekday = now.strftime("%A")
        pred_anchor = night_anchor_date(now)
        # Match the historical gaps' weekday convention (night-anchor of the gap
        # start), not now's calendar day, so the partial pooling is consistent.
        tail_weekday = night_anchor_date(last_tweet).strftime("%A")
        tail_info = tail.predict(now, last_tweet, pred_anchor, tail_weekday)
        comp["p_term"] = tail_info["p_until_am"]
        comp["term_lo"], comp["term_hi"] = tail_info["p_until_lo"], tail_info["p_until_hi"]
        mc_vals = tail_info["samples_s"]
        mc_wts = np.ones(len(mc_vals))
        mc_tau = tail_info["tau"]
    elif out_of_night:
        mode = "daytime"
        p_sleep, lo, hi = model.marginal_p_asleep(cur_s)
        comp["p_term"], comp["term_lo"], comp["term_hi"] = model.marginal_p_terminal(cur_s)
        comp["delta_wd"] = 0.0
        weekday = now.strftime("%A")
        mc_vals, mc_wts, mc_tau, pred_anchor = model._daytime_predictive_pool(
            last_s, silence_min, cur_s, anchor, night_anchor_date(now))
    else:
        mode = "in_night"
        pred_anchor = anchor
        mc_vals, mc_wts, mc_tau = model._predictive_pool(
            weekday, last_s, silence_min, act_dur)

    target_s = _resolve_target_s(pred_anchor, NEXT_TWEET_TARGET)
    # Bootstrap CI is meaningful only for a HISTORICAL pool; the tail pool is
    # model-generated draws (uncertainty already in their spread), so skip it.
    mc = model.next_tweet_montecarlo(mc_vals, mc_wts, mc_tau, target_s=target_s,
                                     epistemic=(mode != "tail"))

    # Legacy next-tweet fields: in-night uses the kernel quantiles / two-branch
    # expectation; the other regimes derive them from the active MC predictor in
    # its own anchor frame so the monitor CSV never logs the stale (empty-pool)
    # in-night quantiles.
    if mode != "in_night" and mc is not None:
        q_next_s = {q: mc["quantiles"][q] for q in PRED_QUANTILES}
        q_next_neff = float(mc["pool_weight"])
        legacy_anchor = pred_anchor
        exp_s = awk_s = wk_s = mc["mean_s"]
    else:
        q_next_s, q_next_neff = q_s, q_neff
        legacy_anchor = anchor
        exp_s, awk_s, wk_s = combined_s, awake_s, wake_s

    return {
        "mc": mc, "mc_anchor": pred_anchor, "mode": mode, "tail": tail_info,
        "now": now, "last_tweet": last_tweet,
        "silence_min": silence_raw, "silence_eff_min": silence_min,
        "weekday": weekday, "anchor": anchor, "last_s": last_s, "cur_s": cur_s,
        "out_of_night": out_of_night,
        "regime": "NIGHT" if in_zone else "DAY",
        "p_asleep_now": p_sleep,                         # (#6) no pin
        "p_terminal": comp["p_term"],                    # until-morning event
        "p_term_lo": comp["term_lo"], "p_term_hi": comp["term_hi"],
        "p_lo": lo, "p_hi": hi, "components": comp,
        "w_sleep": sel["w_sleep"], "act_sleep": act_sleep, "tier_sleep": comp["tier"],
        "w_dur": sel["w_dur"], "act_dur": act_dur, "tier_dur": tier_dur,
        "expected_next_s": exp_s, "awake_exp_s": awk_s, "wake_exp_s": wk_s,
        "expected_next_dt": session_to_datetime(legacy_anchor, exp_s),
        "awake_exp_dt": session_to_datetime(legacy_anchor, awk_s),
        "wake_exp_dt": session_to_datetime(legacy_anchor, wk_s),
        "q_next_s": q_next_s, "q_next_neff": q_next_neff,
        "q_next_dt": {q: session_to_datetime(legacy_anchor, s) for q, s in q_next_s.items()},
    }


def fmt_session_hour(s):
    return decimal_to_time_str(s % 24)

def fmt_minutes(m):
    if m is None:
        return ">8h"
    h, mm = divmod(int(m), 60)
    return f"{h}h{mm:02d}m" if h else f"{mm}m"


def print_montecarlo_section(state, f):
    """Render the Monte Carlo next-tweet predictive (uncertainty + the
    probability the next tweet lands at the target time)."""
    mc = state.get("mc")
    print(f"\n  ── NEXT TWEET — MONTE CARLO PREDICTIVE (v4) ─────────────────", file=f)
    if mc is None:
        print(f"  Insufficient matched history to simulate the next tweet.", file=f)
        return

    anchor = state["mc_anchor"]
    def dt(s, fmt="%I:%M %p EST (%a %m/%d)"):
        return session_to_datetime(anchor, s).strftime(fmt)
    q = mc["quantiles"]

    boot_lbl = (f"{mc['boot_n']} bootstrap resamples" if state.get("mode") != "tail"
                else "EVT posterior-predictive draws")
    print(f"  {mc['mc_n']:,} draws · ±{mc['jitter_min']:.0f}m timing jitter · "
          f"{boot_lbl} · pool n={mc['n_pool']}", file=f)
    print(f"  Predicted next tweet : {dt(q[0.50])}   [median]", file=f)
    print(f"  50% interval         : {dt(q[0.25], '%I:%M %p')} – {dt(q[0.75])}", file=f)
    print(f"  90% interval         : {dt(q[0.05], '%I:%M %p')} – {dt(q[0.95])}"
          f"   (5th–95th pct)", file=f)

    tgt = mc["target_s"]
    tgt_lbl = ("the predicted median" if not NEXT_TWEET_TARGET else "the target")
    print(f"\n  P(next tweet lands near {dt(tgt, '%I:%M %p EST')} — {tgt_lbl}):", file=f)
    for w in sorted(mc["window_probs"]):
        pt, lo, hi = mc["window_probs"][w]
        ci = f"   (95% CI {lo:.1%}–{hi:.1%})" if lo is not None else ""
        print(f"    within ±{w:>3d} min : {pt:6.1%}{ci}", file=f)
    print(f"  P(next tweet before {dt(tgt, '%I:%M %p')}) : {mc['p_before_target']:.1%}", file=f)

    print(f"\n  Hour-by-hour probability of the next tweet:", file=f)
    for hs, pm in mc["hour_buckets"]:
        bar = "█" * int(round(pm * 40))
        print(f"    {dt(hs, '%I:%M %p')} {pm:6.1%} {bar}", file=f)
    ph, pp = mc["peak_bucket"]
    print(f"  Most likely hour     : {dt(ph, '%I:%M %p EST')} ({pp:.1%})", file=f)


def sleep_state_inference(tweet_times, f):
    gap_obs = build_gap_observations(tweet_times)
    model = SleepStateModel(gap_obs)

    print("\n" + "=" * 70, file=f)
    print("  SLEEP-STATE INFERENCE v4 — P(asleep | last tweet time, silence)", file=f)
    print("=" * 70, file=f)
    n_awake = len(model.awake)
    n_term = len(model.terminal)
    n_short = int((gap_obs["kind"] == "shortnight").sum())
    print(f"\n  Training data: {n_awake} awake gaps, {n_term} terminal (until-morning)", file=f)
    print(f"  gaps, {n_short} shortnight gaps (≥{SLEEP_GAP_HOURS:.0f}h, bedtime-window start, ended", file=f)
    print(f"  before {decimal_to_time_str(NIGHT_END_SESSION_H % 24)}). P(asleep) counts terminal+shortnight as sleep;", file=f)
    print(f"  the until-morning event counts terminal only (#1).", file=f)
    print(f"  Shrinkage κ tuned by contiguous-block CV: κ = {model.kappa:.0f}", file=f)
    print(f"  Weekday = night's starting day ('Monday' = Mon evening → Tue morning);", file=f)
    print(f"  weekday enters as a log-odds main effect per regime side (#8).", file=f)

    # ── Per-weekday conditional probability + next-tweet median tables ──
    # Tables show the UNTIL-MORNING event (no more tweets before 5 AM): once a
    # night silence reaches 3h, "in a night-rest gap" is certain by definition,
    # so the until-morning probability is the informative quantity here.
    print(f"\n  (Note: after {SLEEP_GAP_HOURS:.0f}h of silence following a bedtime-window tweet,", file=f)
    print(f"  'asleep now' is certain by definition — tables therefore show the", file=f)
    print(f"  non-degenerate until-morning event probability.)", file=f)
    for day in DAYS_ORDER:
        if not (gap_obs["weekday"] == day).any():
            continue
        header = "  Last tweet  " + "".join(f"{fmt_minutes(m):>8}" for m in SCENARIO_SILENCE_MIN)

        print(f"\n  ── {day.upper()} — P(no more tweets until 5 AM | last tweet, silence) ──", file=f)
        print(header, file=f)
        print("  " + "-" * (len(header) - 2), file=f)
        for ls in SCENARIO_LAST_HOURS:
            cells = []
            for m in SCENARIO_SILENCE_MIN:
                _, _, _, comp = model.p_asleep(day, ls, m)
                cells.append(f"{comp['p_term']:>7.0%} ")
            print(f"  {fmt_session_hour(ls)[:8]:<12}" + "".join(cells), file=f)

        print(f"\n     {day} — MEDIAN time of next tweet | same evidence (#5)", file=f)
        wide = "  Last tweet  " + "".join(f"{fmt_minutes(m):>10}" for m in SCENARIO_SILENCE_MIN)
        print(wide, file=f)
        print("  " + "-" * (len(wide) - 2), file=f)
        for ls in SCENARIO_LAST_HOURS:
            cells = []
            for m in SCENARIO_SILENCE_MIN:
                qd, _ = model.next_tweet_quantiles(day, ls, m)
                cells.append(f"{fmt_session_hour(qd[0.5])[:8]:>10}")
            print(f"  {fmt_session_hour(ls)[:8]:<12}" + "".join(cells), file=f)

    # ── Data-derived confirmation thresholds ──
    print("\n" + "=" * 70, file=f)
    print("  OPTIMAL DOWN-FOR-THE-NIGHT CONFIRMATION THRESHOLDS (per weekday)", file=f)
    print("  Minimal silence after a tweet at hour H for", file=f)
    print("  P(no more tweets until 5 AM) ≥ target.", file=f)
    print("  '*' = the LOWER 95% credible bound also clears the target at that", file=f)
    print("  silence (robust confirmation; #4).", file=f)
    print("=" * 70, file=f)
    for day in DAYS_ORDER:
        if not (gap_obs["weekday"] == day).any():
            continue
        print(f"\n  {day}:", file=f)
        print(f"  {'Last tweet':<12}" + "".join(f"{'≥'+format(t,'.0%'):>10}" for t in CONFIRM_TARGETS), file=f)
        for ls in SCENARIO_LAST_HOURS:
            cells = []
            for target in CONFIRM_TARGETS:
                found, robust = None, False
                for g in range(15, CONFIRM_SEARCH_MAX_MIN + 1, 15):
                    _, _, _, comp = model.p_asleep(day, ls, g)
                    if comp["p_term"] >= target:
                        found, robust = g, (comp["term_lo"] >= target)
                        break
                cells.append((fmt_minutes(found) + ("*" if robust else "")) if found else ">8h")
            print(f"  {fmt_session_hour(ls)[:8]:<12}" + "".join(f"{c:>10}" for c in cells), file=f)

    # ── Activity covariate: optimal look-back window selection ──
    sel = model.select_activity_windows()
    print("\n" + "=" * 70, file=f)
    print("  ACTIVITY COVARIATE — OPTIMAL LOOK-BACK WINDOW SELECTION (v4)", file=f)
    print("=" * 70, file=f)
    print(f"\n  Contiguous-block {ACTIVITY_CV_FOLDS}-fold CV (#7: adjacent nights share a fold, no", file=f)
    print(f"  interleaving leakage). Adoption rule: mean skill − 1 SE > 0 AND", file=f)
    print(f"  ≥ {ACTIVITY_MIN_FOLDS_POS}/{ACTIVITY_CV_FOLDS} folds positive. Duration scored on AWAKE gaps only.", file=f)
    print(f"\n  {'Window':>8} {'Sleep ΔLL/gap (SE)':>22} {'folds+':>7} "
          f"{'Dur. MSE skill (SE)':>22} {'folds+':>7}", file=f)
    print(f"  {'-'*8} {'-'*22} {'-'*7} {'-'*22} {'-'*7}", file=f)
    for _, r in sel["table"].iterrows():
        mark_s = "*" if (sel["w_sleep"] is not None and int(r["W"]) == sel["w_sleep"]) else " "
        mark_d = "*" if (sel["w_dur"] is not None and int(r["W"]) == sel["w_dur"]) else " "
        print(f"  {int(r['W']):>6}h {r['ll_gain_per_gap']*1000:>+12.2f}"
              f" ({r['ll_se']*1000:.2f})mn{mark_s} {int(r['ll_folds_pos'])}/{ACTIVITY_CV_FOLDS:<5} "
              f"{r['mse_skill']:>+13.2%} ({r['mse_se']:.2%}){mark_d} "
              f"{int(r['mse_folds_pos'])}/{ACTIVITY_CV_FOLDS}", file=f)
    print(f"\n  Selected look-back windows (* above):", file=f)
    print(f"  ├─ Sleep-onset : "
          + (f"{sel['w_sleep']}h" if sel["w_sleep"] is not None
             else "none (failed the adoption rule — covariate not used)"), file=f)
    print(f"  └─ Gap duration: "
          + (f"{sel['w_dur']}h" if sel["w_dur"] is not None
             else "none (failed the adoption rule — covariate not used)"), file=f)

    eff = sel["effects"]
    if "sleep" in eff and eff["sleep"]:
        print(f"\n  Effect of recent activity (night bins, within-bin terciles, "
              f"{sel['w_sleep']}h window):", file=f)
        for t in sorted(eff["sleep"]):
            e = eff["sleep"][t]
            print(f"    {ACTIVITY_TIER_LABELS[t]:<5} activity: P(sleep onset) = "
                  f"{e['p_term']:.1%}  (n={e['n']})", file=f)
    if "dur" in eff and eff["dur"]:
        print(f"\n  Median night awake gap by recent-activity tier ({sel['w_dur']}h window):", file=f)
        for t in sorted(eff["dur"]):
            e = eff["dur"][t]
            print(f"    {ACTIVITY_TIER_LABELS[t]:<5} activity: median gap = "
                  f"{e['median_gap']:.0f} min  (n={e['n']})", file=f)
    print(f"\n  (Scenario tables above are marginal over activity; the live", file=f)
    print(f"  estimate below conditions on it.)", file=f)

    # ── Current-state inference ──
    if SLEEP_INFER_NOW:
        d, h, m = parse_hhmm_or_datetime(SLEEP_INFER_NOW)
        now = EST.localize(datetime.combine(d, datetime.min.time()) + timedelta(hours=h, minutes=m))
    else:
        now = datetime.now(EST)
    if SLEEP_INFER_LAST_TWEET:
        d, h, m = parse_hhmm_or_datetime(SLEEP_INFER_LAST_TWEET)
        last_tweet = pd.Timestamp(EST.localize(datetime.combine(d, datetime.min.time())
                                               + timedelta(hours=h, minutes=m)))
    else:
        last_tweet = tweet_times.iloc[-1]

    state = evaluate_current_state(model, tweet_times, now, last_tweet)
    night_day = state["weekday"]
    comp = state["components"]
    out_of_night = state["out_of_night"]
    mode = state["mode"]

    zone_str = (f"{decimal_to_time_str(SLEEP_ZONE_START_S % 24)[:8]}–"
                f"{decimal_to_time_str(SLEEP_ZONE_END_S % 24)[:8]}")

    print("\n" + "=" * 70, file=f)
    print("  CURRENT SLEEP-STATE ESTIMATE (v4)", file=f)
    print("=" * 70, file=f)
    print(f"\n  Now (EST)            : {now.strftime('%A %Y-%m-%d %I:%M %p')}", file=f)
    print(f"  Last tweet (EST)     : {last_tweet.strftime('%A %Y-%m-%d %I:%M %p')}", file=f)
    print(f"  Silence so far       : {fmt_minutes(state['silence_min'])}"
          + (f" (lag-corrected: {fmt_minutes(state['silence_eff_min'])})"
             if INGEST_LAG_MIN > 0 else ""), file=f)
    if mode != "in_night":
        if mode == "tail":
            past_lbl = "extreme-silence tail regime — see below"
        elif state["regime"] == "NIGHT":
            past_lbl = "silence wrapped into a new night"
        else:
            past_lbl = ("daytime — silence is past the "
                        f"{decimal_to_time_str(NIGHT_OVER_SESSION_H % 24)[:8]} latest-wake boundary")
        print(f"  Current day          : {night_day} ({past_lbl})", file=f)
    else:
        print(f"  Night weekday space  : {night_day} (log-odds offset "
              f"{comp['delta_wd']:+.2f})", file=f)
    print(f"  Clock regime         : {state['regime']} — "
          f"{'inside' if state['regime'] == 'NIGHT' else 'outside'} sleep zone ({zone_str} EST)", file=f)
    if state["act_sleep"] is not None:
        tier_lbl = (ACTIVITY_TIER_LABELS[state["tier_sleep"]]
                    if state["tier_sleep"] is not None else "n/a (sparse hour bin)")
        print(f"  Recent activity      : {state['act_sleep']} tweets in the "
              f"{state['w_sleep']}h before the last tweet → tier {tier_lbl}", file=f)
    if state["act_dur"] is not None and state["w_dur"] != state["w_sleep"]:
        tier_lbl = (ACTIVITY_TIER_LABELS[state["tier_dur"]]
                    if state["tier_dur"] is not None else "n/a (sparse hour bin)")
        print(f"  Activity (dur.)      : {state['act_dur']} tweets in the "
              f"{state['w_dur']}h before the last tweet → tier {tier_lbl}", file=f)

    if state["silence_min"] < 0:
        print("\n  Last tweet is in the future relative to 'now' — check the overrides.", file=f)
        return

    print(f"\n  ── POSTERIOR ─────────────────────────────────────────────────", file=f)
    if mode == "tail":
        # The silence is in the deseasonalized extreme tail: conditioning on the
        # last tweet is impossible (off the support), so the EVT residual-life
        # model governs the next tweet and P(until 5 AM); P(asleep) stays the
        # marginal clock-hour state prior.
        ti = state["tail"]
        shape_note = ("bounded tail — finite ceiling" if ti["xi_med"] < -1e-3 else
                      "heavy tail" if ti["xi_med"] > 1e-3 else "exponential tail")
        print(f"  ⚠ EXTREME-SILENCE TAIL REGIME (EVT / Peaks-Over-Threshold GPD)", file=f)
        print(f"  The {fmt_minutes(state['silence_min'])} silence is in the deseasonalized tail:", file=f)
        print(f"  rescaled elapsed {ti['E']:.1f} 'expected tweets missed' vs threshold "
              f"{ti['u']:.1f} and record {ti['rgap_max']:.1f} (percentile {ti['E_pct']:.1%}).", file=f)
        print(f"  GPD shape ξ = {ti['xi_med']:+.2f} [90% {ti['xi_lo']:+.2f}, {ti['xi_hi']:+.2f}] "
              f"from {ti['n_exceed']} exceedances — {shape_note}.", file=f)
        if ti["regime_change"]:
            print(f"  ⚠ PAST THE HISTORICAL MAXIMUM — possible regime change (dormant/", file=f)
            print(f"    offline). Competing-risk weight p(dormant)={ti['p_dormant']:.0%}; intervals", file=f)
            print(f"    widened. Confirm the account is still active before trusting a point.", file=f)
        print(f"  Median additional silence (residual-life): ≈ {fmt_minutes(ti['resid_median_h'] * 60)}", file=f)
        print(f"  P(ASLEEP — in a night-rest gap) = {state['p_asleep_now']:.1%}   "
              f"(95% band: {state['p_lo']:.1%} – {state['p_hi']:.1%}; marginal state prior)", file=f)
        print(f"  P(AWAKE)                        = {1 - state['p_asleep_now']:.1%}", file=f)
        print(f"  P(no more tweets until {decimal_to_time_str(NIGHT_END_SESSION_H % 24)[:8]}) = "
              f"{state['p_terminal']:.1%}   "
              f"(95% band: {state['p_term_lo']:.1%} – {state['p_term_hi']:.1%}; EVT survival)", file=f)
    elif out_of_night:
        # The silence has run past the latest modeled wake time, so the last
        # tweet can no longer be conditioned on (empty evidence pool). We fall
        # back to the marginal occupancy P(asleep) at the current clock hour:
        # ~0% in the daytime, but realistic if the silence has wrapped into a
        # new night. The Monte Carlo predictor below simulates a fresh gap.
        print(f"  Silence has run {fmt_minutes(state['silence_min'])} past the last tweet at "
              f"{last_tweet.strftime('%I:%M %p')}, beyond the", file=f)
        print(f"  {decimal_to_time_str(NIGHT_OVER_SESSION_H % 24)[:8]} latest-wake boundary, so the last tweet can no longer be", file=f)
        print(f"  conditioned on. Falling back to the historical rate of being asleep", file=f)
        print(f"  at this clock hour ({decimal_to_time_str(state['cur_s'] % 24)[:8]}):", file=f)
        print(f"  P(ASLEEP — in a night-rest gap) = {state['p_asleep_now']:.1%}   "
              f"(95% band: {state['p_lo']:.1%} – {state['p_hi']:.1%}; marginal at this hour)", file=f)
        print(f"  P(AWAKE)                        = {1 - state['p_asleep_now']:.1%}", file=f)
        print(f"  P(no more tweets until {decimal_to_time_str(NIGHT_END_SESSION_H % 24)[:8]}) = "
              f"{state['p_terminal']:.1%}   "
              f"(95% band: {state['p_term_lo']:.1%} – {state['p_term_hi']:.1%}; marginal at this hour)", file=f)
    else:
        asleep_k, match_n = direct_empirical_check(tweet_times, night_day,
                                                   state["last_s"], state["silence_eff_min"])

        print(f"  P(ASLEEP — in a night-rest gap) = {state['p_asleep_now']:.1%}   "
              f"(95% band: {state['p_lo']:.1%} – {state['p_hi']:.1%})", file=f)
        print(f"  P(AWAKE)                        = {1 - state['p_asleep_now']:.1%}", file=f)
        print(f"  P(no more tweets until {decimal_to_time_str(NIGHT_END_SESSION_H % 24)[:8]}) = "
              f"{state['p_terminal']:.1%}   "
              f"(95% band: {state['p_term_lo']:.1%} – {state['p_term_hi']:.1%})", file=f)
        if state["regime"] == "DAY":
            print(f"  (Daytime regime: the bedtime prior keeps these naturally small —", file=f)
            print(f"  no hard pin, no discontinuity at the 10 PM boundary; #6.)", file=f)
        print(f"\n  Evidence-matched gaps (started within ±{int(EVIDENCE_S0_TOL_H*60)}m of "
              f"{fmt_session_hour(state['last_s'])[:8]},", file=f)
        print(f"  silent ≥ {fmt_minutes(state['silence_eff_min'])}) — kernel-weighted share that ended the night:", file=f)
        if comp["tier"] is not None:
            print(f"  ├─ {'Same activity tier':<24}: k={comp['k_wd']}, n={comp['n_wd']}"
                  f" (shrunk: {comp['p_tier']:.1%})", file=f)
        print(f"  ├─ {'All weekdays, kernel':<24}: k={comp['k_pool']}, n={comp['n_pool']}"
              f" (shrunk: {comp['p_pool']:.1%})", file=f)
        print(f"  ├─ {'Base rate (same regime)':<24}: {comp['p_base']:.1%} (n={comp['n_base']})", file=f)
        print(f"  └─ {'Weekday log-odds offset':<24}: {comp['delta_wd']:+.2f} ({night_day})", file=f)
        if match_n > 0:
            e_lo, e_hi = jeffreys_interval(asleep_k, match_n)
            print(f"\n  Direct empirical cross-check (model-free): on {asleep_k}/{match_n} similar", file=f)
            print(f"  historical {night_day} nights the silence lasted until morning", file=f)
            print(f"  ({asleep_k/match_n:.0%}, Jeffreys 95% CI {e_lo:.0%}–{e_hi:.0%}).", file=f)
        else:
            print(f"\n  Direct empirical cross-check: no historical {night_day} night matched", file=f)
            print(f"  this evidence pattern within ±45 min.", file=f)

    # ── Monte Carlo next-tweet predictive — shown in all regimes ──
    print_montecarlo_section(state, f)

    if mode == "in_night":
        print(f"\n  Secondary two-branch expectation:", file=f)
        print(f"  ├─ If he tweets again before the morning : "
              f"≈ {state['awake_exp_dt'].strftime('%I:%M %p EST (%a %m/%d)')}", file=f)
        print(f"  ├─ If the silence lasts until the morning: "
              f"≈ {state['wake_exp_dt'].strftime('%I:%M %p EST (%a %m/%d)')}"
              f"  [bedtime-conditioned; #3]", file=f)
        print(f"  └─ Probability-weighted mean             : "
              f"{state['expected_next_dt'].strftime('%I:%M %p EST (%a %m/%d)')}", file=f)
    print("\n" + "=" * 70 + "\n", file=f)


# ── Monitor Mode ─────────────────────────────────────────────────────────────
MONITOR_LOG_COLUMNS = [
    "logged_at_est", "regime", "weekday_space", "last_tweet_est", "silence_min",
    "p_asleep_now", "p_asleep_lo", "p_asleep_hi",
    "p_until_morning", "p_until_morning_lo", "p_until_morning_hi",
    "next_q25_est", "next_median_est", "next_q75_est", "next_q90_est",
    "expected_next_mean_est", "expected_if_tweets_tonight_est",
    "expected_if_silent_until_morning_est", "pred_pool_weight",
    "n_kernel", "act_window_h", "act_count", "act_tier",
    "act_dur_window_h", "act_dur_count", "act_dur_tier",
    "mc_median_est", "mc_q05_est", "mc_q95_est", "mc_target_est",
    "mc_p_within_30m", "mc_peak_hour_est", "mc_peak_prob", "mc_pool_n",
    "mode", "tail_xi", "tail_resid_h", "regime_change",
]

def _tier_label(t):
    return ACTIVITY_TIER_LABELS[t] if t is not None else ""

def _mc_row_fields(state):
    """Monte Carlo fields for the monitor log (blank when MC is unavailable)."""
    mc = state.get("mc")
    if mc is None:
        return ["", "", "", "", "", "", "", ""]
    anc = state["mc_anchor"]
    def s2(s):
        return session_to_datetime(anc, s).strftime("%Y-%m-%d %H:%M")
    p30 = mc["window_probs"].get(30, (float("nan"),))[0]
    return [
        s2(mc["quantiles"][0.50]), s2(mc["quantiles"][0.05]), s2(mc["quantiles"][0.95]),
        s2(mc["target_s"]), f"{p30:.4f}",
        s2(mc["peak_bucket"][0]), f"{mc['peak_bucket'][1]:.4f}", mc["n_pool"],
    ]

def _monitor_row(state):
    qd = state["q_next_dt"]
    return [
        state["now"].strftime("%Y-%m-%d %H:%M:%S"),
        state["regime"],
        state["weekday"],
        state["last_tweet"].strftime("%Y-%m-%d %H:%M:%S"),
        f"{state['silence_min']:.1f}",
        f"{state['p_asleep_now']:.4f}",
        f"{state['p_lo']:.4f}",
        f"{state['p_hi']:.4f}",
        f"{state['p_terminal']:.4f}",
        f"{state['p_term_lo']:.4f}",
        f"{state['p_term_hi']:.4f}",
        qd[0.25].strftime("%Y-%m-%d %H:%M"),
        qd[0.5].strftime("%Y-%m-%d %H:%M"),
        qd[0.75].strftime("%Y-%m-%d %H:%M"),
        qd[0.9].strftime("%Y-%m-%d %H:%M"),
        state["expected_next_dt"].strftime("%Y-%m-%d %H:%M"),
        state["awake_exp_dt"].strftime("%Y-%m-%d %H:%M"),
        state["wake_exp_dt"].strftime("%Y-%m-%d %H:%M"),
        f"{state['q_next_neff']:.1f}",
        state["components"]["n_pool"],
        state["w_sleep"] if state["w_sleep"] is not None else "",
        state["act_sleep"] if state["act_sleep"] is not None else "",
        _tier_label(state["tier_sleep"]),
        state["w_dur"] if state["w_dur"] is not None else "",
        state["act_dur"] if state["act_dur"] is not None else "",
        _tier_label(state["tier_dur"]),
        *_mc_row_fields(state),
        state.get("mode", ""),
        f"{state['tail']['xi_med']:.3f}" if state.get("tail") else "",
        f"{state['tail']['resid_median_h']:.2f}" if state.get("tail") else "",
        "1" if (state.get("tail") and state["tail"]["regime_change"]) else "0",
    ]

def flush_monitor_rows(rows):
    """Append queued rows to the log, rotating a file whose schema is stale.
    Raises OSError if the file is locked (e.g. open in Excel) so the caller
    can keep the rows queued and retry next cycle."""
    if os.path.exists(MONITOR_LOG_FILENAME):
        with open(MONITOR_LOG_FILENAME, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first != ",".join(MONITOR_LOG_COLUMNS):
            stamp = datetime.now(EST).strftime("%Y%m%d_%H%M%S")
            os.replace(MONITOR_LOG_FILENAME, f"{MONITOR_LOG_FILENAME}.{stamp}.bak")
    new_file = not os.path.exists(MONITOR_LOG_FILENAME)
    with open(MONITOR_LOG_FILENAME, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(MONITOR_LOG_COLUMNS)
        w.writerows(rows)


def _safe_full_report(label):
    """Run the full analysis (overwriting OUTPUT_FILENAME) without ever
    letting a failure propagate into the monitor loop."""
    try:
        run_analysis()
        print(f"[monitor] {label} full report written to '{OUTPUT_FILENAME}'", flush=True)
    except Exception as e:
        print(f"[monitor] {label} full report failed "
              f"({e.__class__.__name__}: {e}) - monitor continues", flush=True)


def run_monitor():
    """Self-scheduling loop: one evaluation every MONITOR_INTERVAL_MIN minutes,
    logged to MONITOR_LOG_FILENAME. The full analysis report is written once
    at startup and re-written every REPORT_INTERVAL_MIN minutes — both run
    inside this single-threaded loop, so the two schedules cannot conflict.
    Failure tolerance: CSV unreachable/mid-write → reuse last good data;
    shrunken CSV → keep previous data; locked log → queue rows and retry;
    failed report → retried at the next slot; model rebuilt only on change."""
    print(f"Monitor mode (v4): evaluating every {MONITOR_INTERVAL_MIN} min -> "
          f"'{MONITOR_LOG_FILENAME}'; full report on startup and every "
          f"{REPORT_INTERVAL_MIN} min -> '{OUTPUT_FILENAME}' (Ctrl+C to stop; "
          f"use --report for a one-shot full analysis)", flush=True)

    _safe_full_report("startup")
    next_report = datetime.now(EST) + timedelta(minutes=REPORT_INTERVAL_MIN)

    model, tweet_times, n_events = None, None, 0
    pending_rows = []
    try:
        while True:
            cycle_started = time.time()

            now_est = datetime.now(EST)
            if now_est >= next_report:
                _safe_full_report("refresh")
                next_report = now_est + timedelta(minutes=REPORT_INTERVAL_MIN)

            try:
                df = load_data(CSV_FILENAME)
                tt = expand_to_tweet_events(df)
                if len(tt) == 0:
                    raise ValueError("CSV contained no usable tweet rows")
                if n_events and len(tt) < 0.9 * n_events:
                    print(f"[monitor] CSV shrank ({len(tt)} events < {n_events}) - "
                          f"possible partial write; keeping previous data this cycle",
                          flush=True)
                elif (tweet_times is None or len(tt) != n_events
                      or tt.iloc[-1] != tweet_times.iloc[-1]):
                    model = SleepStateModel(build_gap_observations(tt))
                    model.select_activity_windows()
                    tweet_times, n_events = tt, len(tt)
            except Exception as e:
                fallback = ("reusing last good data" if model is not None
                            else "no data yet, retrying next cycle")
                print(f"[monitor] data refresh failed "
                      f"({e.__class__.__name__}: {e}) - {fallback}", flush=True)

            if model is not None:
                try:
                    state = evaluate_current_state(model, tweet_times)
                    if state["silence_min"] < 0:
                        print(f"[{state['now'].strftime('%H:%M:%S')}] last tweet is "
                              f"in the future relative to the clock - skipping cycle",
                              flush=True)
                    else:
                        pending_rows.append(_monitor_row(state))
                        if len(pending_rows) > MONITOR_MAX_QUEUED:
                            pending_rows = pending_rows[-MONITOR_MAX_QUEUED:]
                        act_str = (f"act={state['act_sleep']}@{state['w_sleep']}h"
                                   f"({_tier_label(state['tier_sleep']) or '-'}) "
                                   if state["act_sleep"] is not None else "")
                        mc = state.get("mc")
                        if mc is not None:
                            mc_med = session_to_datetime(state["mc_anchor"],
                                                         mc["quantiles"][0.5])
                            p30 = mc["window_probs"].get(30, (float("nan"),))[0]
                            mc_str = (f"MC median~{mc_med.strftime('%I:%M %p %m/%d')} "
                                      f"P(±30m)={p30:.0%}")
                        else:
                            mc_str = "MC n/a"
                        print(f"[{state['now'].strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"{state['regime']:<5} {state['weekday']:<9} "
                              f"silence={fmt_minutes(state['silence_min'])} {act_str}"
                              f"P(asleep)={state['p_asleep_now']:.0%} "
                              f"P(until-am)={state['p_terminal']:.0%} "
                              f"{mc_str}",
                              flush=True)
                except Exception as e:
                    print(f"[monitor] evaluation failed "
                          f"({e.__class__.__name__}: {e})", flush=True)

            if pending_rows:
                try:
                    flush_monitor_rows(pending_rows)
                    pending_rows = []
                except OSError as e:
                    print(f"[monitor] log write failed ({e}) - "
                          f"{len(pending_rows)} row(s) queued for retry", flush=True)

            elapsed = time.time() - cycle_started
            time.sleep(max(1.0, MONITOR_INTERVAL_MIN * 60 - elapsed))
    except KeyboardInterrupt:
        if pending_rows:
            try:
                flush_monitor_rows(pending_rows)
            except OSError:
                print(f"[monitor] could not flush {len(pending_rows)} queued "
                      f"row(s) on exit", flush=True)
        print("\nMonitor stopped.")


# ── Main Execution ──────────────────────────────────────────────────────────
def run_analysis():
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            print("=" * 70)
            print("  Twitter Sleep / Wake Pattern Analysis (v4)")
            print("=" * 70)

            df = load_data(CSV_FILENAME)
            tweet_times = expand_to_tweet_events(df)
            sleep_periods = find_sleep_periods(tweet_times)
            launch_dates = get_launch_dates_est()

            if not sleep_periods:
                print("No sleep periods detected.")
                return

            for sp in sleep_periods:
                sp["bedtime_tweets"] = count_session_tweets(tweet_times, sp["sleep_start"], BEDTIME_SESSION_MIN, "before")
                sp["morning_tweets"] = count_session_tweets(tweet_times, sp["wake_time"], MORNING_SESSION_MIN, "after")
                sp["wake_hour"] = sp["wake_time"].hour + sp["wake_time"].minute / 60.0

            sp_df = pd.DataFrame(sleep_periods)
            sp_df["is_launch_day"] = sp_df["date_label"].isin(launch_dates)

            sleep_analysis_subset(sp_df[sp_df["is_launch_day"]].copy(), "LAUNCH DAYS", f)
            sleep_analysis_subset(sp_df[~sp_df["is_launch_day"]].copy(), "NON-LAUNCH DAYS", f)

            sleep_state_inference(tweet_times, f)

        finally:
            sys.stdout = original_stdout

    try:
        print(f"✅ Analysis complete. All output saved to '{OUTPUT_FILENAME}'")
    except UnicodeEncodeError:
        print(f"Analysis complete. All output saved to '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    # Scheduling lives in the script: the default is the self-repeating
    # monitor loop. --report (or --once) produces the one-shot full analysis.
    if "--report" in sys.argv or "--once" in sys.argv:
        run_analysis()
    else:
        run_monitor()
