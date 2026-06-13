"""Sleep/wake conditional probability + next-tweet prediction - v2.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import beta as beta_dist
import pytz
import sys
import os
import csv
import time

# == Configuration ============================================================
_DATA           = os.environ.get("DATA_DIR", ".")
CSV_FILENAME    = os.path.join(_DATA, "elonmusk_tweet_history.csv")
OUTPUT_FILENAME = os.path.join(_DATA, "final_output_v2.txt")

SLEEP_GAP_HOURS      = 3.0
EARLIEST_BEDTIME_H   = 22
LATEST_BEDTIME_H     = 31
EARLIEST_WAKE_H      = 8
LATEST_WAKE_H        = 13
BEDTIME_SESSION_MIN  = 90
MORNING_SESSION_MIN  = 90
ANOMALY_GAP_MULT     = 2.5
EST                  = pytz.timezone("America/New_York")

# == Sleep-State Inference Configuration ======================================
SLEEP_INFER_NOW        = None    # "6/12/2026 02:30" (EST) or None -> current clock time
SLEEP_INFER_LAST_TWEET = None    # "6/12/2026 01:10" (EST) or None -> most recent tweet in CSV
SHRINK_KAPPA           = 8.0     # fallback shrinkage strength; v2 tunes kappa by CV (#8)
KAPPA_GRID             = [2.0, 4.0, 8.0, 16.0, 32.0]
SCENARIO_LAST_HOURS    = [22, 23, 24, 25, 26, 27]   # last-tweet clock: 10PM..3AM (session scale)
SCENARIO_SILENCE_MIN   = [30, 60, 90, 120, 180, 240, 300, 360]
CONFIRM_TARGETS        = [0.80, 0.90, 0.95]
CONFIRM_SEARCH_MAX_MIN = 480     # search silence thresholds up to 8 h
DIRECT_MATCH_TOL_H     = 0.75    # +/-45 min last-tweet-time tolerance for empirical cross-check
NIGHT_END_SESSION_H    = 29      # next tweet at/after 5 AM (session hour 29) ends the night
MAX_WAKE_SESSION_H     = 38      # ignore multi-day disappearances when modelling wake times

# v2 evidence matching (#2): centered kernel on the last-tweet clock time
# instead of floor-hour bins, widened until the pool is adequately populated.
EVIDENCE_S0_TOL_H      = 0.75            # base kernel half-width (hours)
EVIDENCE_TOL_WIDEN     = [0.75, 1.5, 3.0]
WAKE_BEDTIME_TOL_H     = 1.5             # (#3) wake distribution conditioned on bedtime +/- this
WEEKDAY_POOL_WEIGHT    = 2.0             # (#5) same-weekday rows up-weighted in quantile pools
TIER_POOL_WEIGHT       = 1.5             # (#5) matching-activity-tier rows up-weighted
PRED_QUANTILES         = [0.25, 0.50, 0.75, 0.90]

# Live silence is measured from the CSV's last row; if the updater polls every
# N minutes, observed silence carries a +N/2 ingestion-lag bias. Set this to
# the updater's mean latency (minutes) to subtract it from live silences.
INGEST_LAG_MIN         = 0.0

# Sleep zone (used for the regime-matched base rate and display; v2 no longer
# pins P(asleep)=0 outside it - see #6).
SLEEP_ZONE_START_S     = float(EARLIEST_BEDTIME_H)   # 22.0 -> 10 PM EST
SLEEP_ZONE_END_S       = float(LATEST_WAKE_H + 24)   # 37.0 -> 1 PM EST next day

# == Monitor Mode Configuration ===============================================
MONITOR_INTERVAL_MIN   = 1
MONITOR_LOG_FILENAME   = os.path.join(_DATA, "next_tweet_monitor_log_v2.csv")
MONITOR_MAX_QUEUED     = 500
DAILY_REPORT_HOUR_EST  = 22

# == Activity Covariate Configuration =========================================
ACTIVITY_WINDOWS_H     = [1, 2, 3, 4, 6, 8, 12, 24]
ACTIVITY_TIER_LABELS   = ["LOW", "MID", "HIGH"]
ACTIVITY_CV_FOLDS      = 5
ACTIVITY_MIN_BIN_N     = 30
ACTIVITY_MIN_FOLDS_POS = 4    # (#7) adoption also needs >= this many positive folds

# == Launch Day Timestamps =====================================================
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

# == Helpers ==================================================================
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

# == Core Data Functions =====================================================
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

# == Sleep Summary ============================================================
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

        print(f"\n  += {day.upper()} ({n} nights) ========================================", file=file)
        print(f"  | Avg bedtime : {avg_sleep_str} | Avg wake-up : {avg_wake_str}", file=file)
        print(f"  | Earliest bed : {fmt_time(earliest_sleep)} ({earliest_sleep.strftime('%Y-%m-%d')}) | "
              f"Latest bed : {fmt_time(latest_sleep)} ({latest_sleep.strftime('%Y-%m-%d')})", file=file)
        print(f"  | Earliest wake : {fmt_time(earliest_wake)} ({earliest_wake.strftime('%Y-%m-%d')}) | "
              f"Latest wake : {fmt_time(latest_wake)} ({latest_wake.strftime('%Y-%m-%d')})", file=file)
        print(f"  | Sleep duration: avg={avg_sleep_dur:.1f}h min={min_sleep_dur:.1f}h max={max_sleep_dur:.1f}h", file=file)
        print(f"  | Bed session : avg={bt.mean():.1f} tweets P(<5)={p_bed_lt5:.1%}", file=file)
        print(f"  | Morn session : avg={mt.mean():.1f} tweets P(<5)={p_morn_lt5:.1%}", file=file)
        print(f"  +========================================================", file=file)


# == Sleep-State Inference (v2) ===============================================
#
# Evidence at query time tau: last tweet at tau0 = tau - g, silent for g hours.
# Every historical inter-tweet gap is one trial. Classes (#1) - all three are
# OBSERVABLE (no label noise):
#   terminal   - gap >= SLEEP_GAP_HOURS and ends at/after NIGHT_END_SESSION_H
#                (5 AM): the silence demonstrably lasted until morning.
#   shortnight - gap >= SLEEP_GAP_HOURS starting in the bedtime window but
#                ending BEFORE 5 AM: a short night sleep that ended early.
#   awake      - everything else.
#
# Two posteriors are estimated from the same evidence pool (#1):
#   p_until_morning - P(terminal): the betting-relevant "no more tweets until
#                     5 AM" event; shortnight gaps are real negatives here.
#   p_asleep        - P(terminal or shortnight): "currently in a night-rest
#                     gap"; the sleep-state interpretation.
# Estimation is by direct conditioning (#2): among gaps that started within a
# +/-EVIDENCE_S0_TOL_H kernel of tau0's clock time AND lasted at least g, the
# (kernel-weighted) positive share. Shrinkage chain: regime base rate ->
# kernel pool -> activity tier, each with kappa pseudo-counts (kappa tuned by grouped
# CV, #8). The weekday enters as a log-odds main effect estimated from all
# gaps on the same regime side (#8), and the credible band is the Beta
# posterior implied by the shrinkage (#4).
#
# The predictive distribution of the next-tweet time is read directly off the
# end times of the evidence-matched gaps (weighted quantiles, #5); the
# two-branch mixture expectation is kept as a secondary diagnostic with the
# wake branch conditioned on bedtime proximity (#3).
#
# Session-hour scale: anchored at 1 PM EST so one night (13:00 -> 12:59 next
# day) is contiguous: 22 = 10 PM, 24 = midnight, 26 = 2 AM, 33 = 9 AM.
# Kernels do not wrap across the 1 PM boundary (negligible: lowest-stakes hour
# of the cycle). end_s = s0 + elapsed drifts +/-1h from clock time on the two
# DST nights per year - accepted.

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

    # == shrinkage / fold helpers =============================================
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
        terminal model - the same shrinkage structure the live estimator
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

    # == evidence pools (#2) ==================================================
    def _evidence_pool(self, last_s, silence_min, frame=None, tol=EVIDENCE_S0_TOL_H):
        """Gaps matching the evidence: started within +/-tol (session hours) of
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

    # == Activity covariate machinery ========================================
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
        duration scored on AWAKE gaps only - the deployed quantity (#7)."""
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

                # - sleep-onset models -
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

                # - duration models: awake gaps only (#7) -
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

    # == posterior (#1, #2, #4, #8) ===========================================
    def p_asleep(self, weekday, last_s, silence_min, activity_count=None):
        """Two posteriors from one kernel evidence pool (#1):
          primary return - P(asleep: currently in a night-rest gap), i.e.
                           positive = terminal OR shortnight;
          components['p_term'] (+band) - P(no more tweets until 5 AM), i.e.
                           positive = terminal only (the betting event).
        Pool conditioned on the observed silence (#2), kappa-shrunk through
        regime base -> pool -> activity tier; weekday applied as a log-odds
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

    # == predictive distribution of the next tweet (#5) ======================
    def next_tweet_quantiles(self, weekday, last_s, silence_min, act_dur=None):
        """Weighted quantiles of the next-tweet session-hour, directly from
        end times of evidence-matched gaps. Weights: triangular kernel in
        start-time distance x WEEKDAY_POOL_WEIGHT for same-weekday rows x
        TIER_POOL_WEIGHT for matching-activity-tier rows. The kernel widens
        until the pool is adequately populated; final fallback is the
        bedtime-agnostic wake distribution."""
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
                return {q: tau + 0.5 for q in PRED_QUANTILES}, 0.0
            vals, wts = wk["end_s"].values, np.ones(len(wk))
        else:
            vals = pool["end_s"].values
            wts = w.values.copy()
            wts = wts * np.where(pool["weekday"].values == weekday, WEEKDAY_POOL_WEIGHT, 1.0)
            if tier_d is not None and "tier_dur" in pool.columns:
                wts = wts * np.where(pool["tier_dur"].values == float(tier_d),
                                     TIER_POOL_WEIGHT, 1.0)
        qs = _weighted_quantiles(vals, wts, PRED_QUANTILES)
        return dict(zip(PRED_QUANTILES, qs)), float(np.sum(wts))

    # == two-branch expectation (secondary diagnostic; #3 wake conditioning) ==
    def expected_next_tweet(self, weekday, last_s, silence_min,
                            act_sleep=None, act_dur=None):
        """Posterior-weighted expected session-hour of the next tweet. The
        sleep branch (terminal + shortnight ends) conditions on bedtime
        proximity (+/-WAKE_BEDTIME_TOL_H, widening; #3); the awake branch uses
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


def direct_empirical_check(tweet_times, weekday, last_s, silence_min):
    """Model-free cross-check: among historical nights of this weekday where
    the last tweet fell within +/-45 min of the same clock time and silence had
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
    absolute EST datetime: 26.5 on anchor 2026-06-11 -> 2026-06-12 02:30 EST."""
    naive = datetime.combine(anchor_date, datetime.min.time()) + timedelta(hours=s)
    return EST.localize(naive)


def count_recent_tweets(tweet_times, t_end, hours):
    """Tweets (burst-inclusive event counts) in (t_end - hours, t_end]."""
    arr = np.sort(tweet_times.values.astype("datetime64[ns]"))
    t1 = pd.Timestamp(t_end).to_datetime64()
    hi = np.searchsorted(arr, t1, side="right")
    lo = np.searchsorted(arr, t1 - np.timedelta64(int(hours * 60), "m"), side="right")
    return int(hi - lo)


def evaluate_current_state(model, tweet_times, now=None, last_tweet=None):
    """Single source of truth for the live estimate. v2 (#6): no hard daytime
    pin - p_terminal (P(no more tweets until tomorrow morning)) is reported in
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
    in_zone = cur_s >= SLEEP_ZONE_START_S

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

    return {
        "now": now, "last_tweet": last_tweet,
        "silence_min": silence_raw, "silence_eff_min": silence_min,
        "weekday": weekday, "anchor": anchor, "last_s": last_s, "cur_s": cur_s,
        "regime": "NIGHT" if in_zone else "DAY",
        "p_asleep_now": p_sleep,                         # (#6) no pin
        "p_terminal": comp["p_term"],                    # until-morning event
        "p_term_lo": comp["term_lo"], "p_term_hi": comp["term_hi"],
        "p_lo": lo, "p_hi": hi, "components": comp,
        "w_sleep": sel["w_sleep"], "act_sleep": act_sleep, "tier_sleep": comp["tier"],
        "w_dur": sel["w_dur"], "act_dur": act_dur, "tier_dur": tier_dur,
        "expected_next_s": combined_s, "awake_exp_s": awake_s, "wake_exp_s": wake_s,
        "expected_next_dt": session_to_datetime(anchor, combined_s),
        "awake_exp_dt": session_to_datetime(anchor, awake_s),
        "wake_exp_dt": session_to_datetime(anchor, wake_s),
        "q_next_s": q_s, "q_next_neff": q_neff,
        "q_next_dt": {q: session_to_datetime(anchor, s) for q, s in q_s.items()},
    }


def fmt_session_hour(s):
    return decimal_to_time_str(s % 24)

def fmt_minutes(m):
    if m is None:
        return ">8h"
    h, mm = divmod(int(m), 60)
    return f"{h}h{mm:02d}m" if h else f"{mm}m"


def sleep_state_inference(tweet_times, f):
    gap_obs = build_gap_observations(tweet_times)
    model = SleepStateModel(gap_obs)

    print("\n" + "=" * 70, file=f)
    print("  SLEEP-STATE INFERENCE v2 - P(asleep | last tweet time, silence)", file=f)
    print("=" * 70, file=f)
    n_awake = len(model.awake)
    n_term = len(model.terminal)
    n_short = int((gap_obs["kind"] == "shortnight").sum())
    print(f"\n  Training data: {n_awake} awake gaps, {n_term} terminal (until-morning)", file=f)
    print(f"  gaps, {n_short} shortnight gaps (>={SLEEP_GAP_HOURS:.0f}h, bedtime-window start, ended", file=f)
    print(f"  before {decimal_to_time_str(NIGHT_END_SESSION_H % 24)}). P(asleep) counts terminal+shortnight as sleep;", file=f)
    print(f"  the until-morning event counts terminal only (#1).", file=f)
    print(f"  Shrinkage kappa tuned by contiguous-block CV: kappa = {model.kappa:.0f}", file=f)
    print(f"  Weekday = night's starting day ('Monday' = Mon evening -> Tue morning);", file=f)
    print(f"  weekday enters as a log-odds main effect per regime side (#8).", file=f)

    # == Per-weekday conditional probability + next-tweet median tables ==
    # Tables show the UNTIL-MORNING event (no more tweets before 5 AM): once a
    # night silence reaches 3h, "in a night-rest gap" is certain by definition,
    # so the until-morning probability is the informative quantity here.
    print(f"\n  (Note: after {SLEEP_GAP_HOURS:.0f}h of silence following a bedtime-window tweet,", file=f)
    print(f"  'asleep now' is certain by definition - tables therefore show the", file=f)
    print(f"  non-degenerate until-morning event probability.)", file=f)
    for day in DAYS_ORDER:
        if not (gap_obs["weekday"] == day).any():
            continue
        header = "  Last tweet  " + "".join(f"{fmt_minutes(m):>8}" for m in SCENARIO_SILENCE_MIN)

        print(f"\n  == {day.upper()} - P(no more tweets until 5 AM | last tweet, silence) ==", file=f)
        print(header, file=f)
        print("  " + "-" * (len(header) - 2), file=f)
        for ls in SCENARIO_LAST_HOURS:
            cells = []
            for m in SCENARIO_SILENCE_MIN:
                _, _, _, comp = model.p_asleep(day, ls, m)
                cells.append(f"{comp['p_term']:>7.0%} ")
            print(f"  {fmt_session_hour(ls)[:8]:<12}" + "".join(cells), file=f)

        print(f"\n     {day} - MEDIAN time of next tweet | same evidence (#5)", file=f)
        wide = "  Last tweet  " + "".join(f"{fmt_minutes(m):>10}" for m in SCENARIO_SILENCE_MIN)
        print(wide, file=f)
        print("  " + "-" * (len(wide) - 2), file=f)
        for ls in SCENARIO_LAST_HOURS:
            cells = []
            for m in SCENARIO_SILENCE_MIN:
                qd, _ = model.next_tweet_quantiles(day, ls, m)
                cells.append(f"{fmt_session_hour(qd[0.5])[:8]:>10}")
            print(f"  {fmt_session_hour(ls)[:8]:<12}" + "".join(cells), file=f)

    # == Data-derived confirmation thresholds ==
    print("\n" + "=" * 70, file=f)
    print("  OPTIMAL DOWN-FOR-THE-NIGHT CONFIRMATION THRESHOLDS (per weekday)", file=f)
    print("  Minimal silence after a tweet at hour H for", file=f)
    print("  P(no more tweets until 5 AM) >= target.", file=f)
    print("  '*' = the LOWER 95% credible bound also clears the target at that", file=f)
    print("  silence (robust confirmation; #4).", file=f)
    print("=" * 70, file=f)
    for day in DAYS_ORDER:
        if not (gap_obs["weekday"] == day).any():
            continue
        print(f"\n  {day}:", file=f)
        print(f"  {'Last tweet':<12}" + "".join(f"{'>='+format(t,'.0%'):>10}" for t in CONFIRM_TARGETS), file=f)
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

    # == Activity covariate: optimal look-back window selection ==
    sel = model.select_activity_windows()
    print("\n" + "=" * 70, file=f)
    print("  ACTIVITY COVARIATE - OPTIMAL LOOK-BACK WINDOW SELECTION (v2)", file=f)
    print("=" * 70, file=f)
    print(f"\n  Contiguous-block {ACTIVITY_CV_FOLDS}-fold CV (#7: adjacent nights share a fold, no", file=f)
    print(f"  interleaving leakage). Adoption rule: mean skill - 1 SE > 0 AND", file=f)
    print(f"  >= {ACTIVITY_MIN_FOLDS_POS}/{ACTIVITY_CV_FOLDS} folds positive. Duration scored on AWAKE gaps only.", file=f)
    print(f"\n  {'Window':>8} {'Sleep DeltaLL/gap (SE)':>22} {'folds+':>7} "
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
    print(f"  += Sleep-onset : "
          + (f"{sel['w_sleep']}h" if sel["w_sleep"] is not None
             else "none (failed the adoption rule - covariate not used)"), file=f)
    print(f"  += Gap duration: "
          + (f"{sel['w_dur']}h" if sel["w_dur"] is not None
             else "none (failed the adoption rule - covariate not used)"), file=f)

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

    # == Current-state inference ==
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

    zone_str = (f"{decimal_to_time_str(SLEEP_ZONE_START_S % 24)[:8]}-"
                f"{decimal_to_time_str(SLEEP_ZONE_END_S % 24)[:8]}")

    print("\n" + "=" * 70, file=f)
    print("  CURRENT SLEEP-STATE ESTIMATE (v2)", file=f)
    print("=" * 70, file=f)
    print(f"\n  Now (EST)            : {now.strftime('%A %Y-%m-%d %I:%M %p')}", file=f)
    print(f"  Last tweet (EST)     : {last_tweet.strftime('%A %Y-%m-%d %I:%M %p')}", file=f)
    print(f"  Silence so far       : {fmt_minutes(state['silence_min'])}"
          + (f" (lag-corrected: {fmt_minutes(state['silence_eff_min'])})"
             if INGEST_LAG_MIN > 0 else ""), file=f)
    print(f"  Night weekday space  : {night_day} (log-odds offset "
          f"{comp['delta_wd']:+.2f})", file=f)
    print(f"  Clock regime         : {state['regime']} - "
          f"{'inside' if state['regime'] == 'NIGHT' else 'outside'} sleep zone ({zone_str} EST)", file=f)
    if state["act_sleep"] is not None:
        tier_lbl = (ACTIVITY_TIER_LABELS[state["tier_sleep"]]
                    if state["tier_sleep"] is not None else "n/a (sparse hour bin)")
        print(f"  Recent activity      : {state['act_sleep']} tweets in the "
              f"{state['w_sleep']}h before the last tweet -> tier {tier_lbl}", file=f)
    if state["act_dur"] is not None and state["w_dur"] != state["w_sleep"]:
        tier_lbl = (ACTIVITY_TIER_LABELS[state["tier_dur"]]
                    if state["tier_dur"] is not None else "n/a (sparse hour bin)")
        print(f"  Activity ({state['w_dur']}h, dur.) : {state['act_dur']} tweets "
              f"-> tier {tier_lbl}", file=f)

    if state["silence_min"] < 0:
        print("\n  Last tweet is in the future relative to 'now' - check the overrides.", file=f)
        return

    asleep_k, match_n = direct_empirical_check(tweet_times, night_day,
                                               state["last_s"], state["silence_eff_min"])

    print(f"\n  == POSTERIOR =================================================", file=f)
    print(f"  P(ASLEEP - in a night-rest gap) = {state['p_asleep_now']:.1%}   "
          f"(95% band: {state['p_lo']:.1%} - {state['p_hi']:.1%})", file=f)
    print(f"  P(AWAKE)                        = {1 - state['p_asleep_now']:.1%}", file=f)
    print(f"  P(no more tweets until {decimal_to_time_str(NIGHT_END_SESSION_H % 24)[:8]}) = "
          f"{state['p_terminal']:.1%}   "
          f"(95% band: {state['p_term_lo']:.1%} - {state['p_term_hi']:.1%})", file=f)
    if state["regime"] == "DAY":
        print(f"  (Daytime regime: the bedtime prior keeps these naturally small -", file=f)
        print(f"  no hard pin, no discontinuity at the 10 PM boundary; #6.)", file=f)
    print(f"\n  Evidence-matched gaps (started within +/-{int(EVIDENCE_S0_TOL_H*60)}m of "
          f"{fmt_session_hour(state['last_s'])[:8]},", file=f)
    print(f"  silent >= {fmt_minutes(state['silence_eff_min'])}) - kernel-weighted share that ended the night:", file=f)
    if comp["tier"] is not None:
        print(f"  += {'Same activity tier':<24}: k={comp['k_wd']}, n={comp['n_wd']}"
              f" (shrunk: {comp['p_tier']:.1%})", file=f)
    print(f"  += {'All weekdays, kernel':<24}: k={comp['k_pool']}, n={comp['n_pool']}"
          f" (shrunk: {comp['p_pool']:.1%})", file=f)
    print(f"  += {'Base rate (same regime)':<24}: {comp['p_base']:.1%} (n={comp['n_base']})", file=f)
    print(f"  += {'Weekday log-odds offset':<24}: {comp['delta_wd']:+.2f} ({night_day})", file=f)
    if match_n > 0:
        e_lo, e_hi = jeffreys_interval(asleep_k, match_n)
        print(f"\n  Direct empirical cross-check (model-free): on {asleep_k}/{match_n} similar", file=f)
        print(f"  historical {night_day} nights the silence lasted until morning", file=f)
        print(f"  ({asleep_k/match_n:.0%}, Jeffreys 95% CI {e_lo:.0%}-{e_hi:.0%}).", file=f)
    else:
        print(f"\n  Direct empirical cross-check: no historical {night_day} night matched", file=f)
        print(f"  this evidence pattern within +/-45 min.", file=f)

    print(f"\n  == NEXT TWEET - PREDICTIVE DISTRIBUTION (#5) =================", file=f)
    qd = state["q_next_dt"]
    print(f"  Median           : {qd[0.5].strftime('%I:%M %p EST (%a %m/%d)')}", file=f)
    print(f"  50% interval     : {qd[0.25].strftime('%I:%M %p')} - "
          f"{qd[0.75].strftime('%I:%M %p EST (%a %m/%d)')}", file=f)
    print(f"  90th percentile  : {qd[0.9].strftime('%I:%M %p EST (%a %m/%d)')}", file=f)
    print(f"  (effective pool weight: {state['q_next_neff']:.1f})", file=f)
    print(f"\n  Secondary two-branch expectation:", file=f)
    print(f"  += If he tweets again before the morning : "
          f"~ {state['awake_exp_dt'].strftime('%I:%M %p EST (%a %m/%d)')}", file=f)
    print(f"  += If the silence lasts until the morning: "
          f"~ {state['wake_exp_dt'].strftime('%I:%M %p EST (%a %m/%d)')}"
          f"  [bedtime-conditioned; #3]", file=f)
    print(f"  += Probability-weighted mean             : "
          f"{state['expected_next_dt'].strftime('%I:%M %p EST (%a %m/%d)')}", file=f)
    print("\n" + "=" * 70 + "\n", file=f)


# == Monitor Mode =============================================================
MONITOR_LOG_COLUMNS = [
    "logged_at_est", "regime", "weekday_space", "last_tweet_est", "silence_min",
    "p_asleep_now", "p_asleep_lo", "p_asleep_hi",
    "p_until_morning", "p_until_morning_lo", "p_until_morning_hi",
    "next_q25_est", "next_median_est", "next_q75_est", "next_q90_est",
    "expected_next_mean_est", "expected_if_tweets_tonight_est",
    "expected_if_silent_until_morning_est", "pred_pool_weight",
    "n_kernel", "act_window_h", "act_count", "act_tier",
    "act_dur_window_h", "act_dur_count", "act_dur_tier",
]

def _tier_label(t):
    return ACTIVITY_TIER_LABELS[t] if t is not None else ""

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


def _next_report_time(after_dt):
    """Next DAILY_REPORT_HOUR_EST occurrence strictly after after_dt (EST,
    DST-safe via per-date localization)."""
    candidate = EST.localize(datetime.combine(after_dt.date(), datetime.min.time())
                             + timedelta(hours=DAILY_REPORT_HOUR_EST))
    if candidate <= after_dt:
        candidate = EST.localize(datetime.combine(after_dt.date() + timedelta(days=1),
                                                  datetime.min.time())
                                 + timedelta(hours=DAILY_REPORT_HOUR_EST))
    return candidate


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
    at startup and re-written daily at DAILY_REPORT_HOUR_EST - both run inside
    this single-threaded loop, so the two schedules cannot conflict.
    Failure tolerance: CSV unreachable/mid-write -> reuse last good data;
    shrunken CSV -> keep previous data; locked log -> queue rows and retry;
    failed report -> retried at the next slot; model rebuilt only on change."""
    print(f"Monitor mode (v2): evaluating every {MONITOR_INTERVAL_MIN} min -> "
          f"'{MONITOR_LOG_FILENAME}'; full report on startup and daily at "
          f"{decimal_to_time_str(DAILY_REPORT_HOUR_EST % 24)} (Ctrl+C to stop; "
          f"use --report for a one-shot full analysis)", flush=True)

    _safe_full_report("startup")
    next_report = _next_report_time(datetime.now(EST))

    model, tweet_times, n_events = None, None, 0
    pending_rows = []
    try:
        while True:
            cycle_started = time.time()

            now_est = datetime.now(EST)
            if now_est >= next_report:
                _safe_full_report("daily")
                next_report = _next_report_time(now_est)

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
                        print(f"[{state['now'].strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"{state['regime']:<5} {state['weekday']:<9} "
                              f"silence={fmt_minutes(state['silence_min'])} {act_str}"
                              f"P(asleep)={state['p_asleep_now']:.0%} "
                              f"P(until-am)={state['p_terminal']:.0%} "
                              f"median~{state['q_next_dt'][0.5].strftime('%I:%M %p %m/%d')}",
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


# == Main Execution ==========================================================
def run_analysis():
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            print("=" * 70)
            print("  Twitter Sleep / Wake Pattern Analysis (v2)")
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
        print(f"Analysis complete. All output saved to '{OUTPUT_FILENAME}'")
    except UnicodeEncodeError:
        print(f"Analysis complete. All output saved to '{OUTPUT_FILENAME}'")


if __name__ == "__main__":
    # Scheduling lives in the script: the default is the self-repeating
    # monitor loop. --report (or --once) produces the one-shot full analysis.
    if "--report" in sys.argv or "--once" in sys.argv:
        run_analysis()
    else:
        run_monitor()
