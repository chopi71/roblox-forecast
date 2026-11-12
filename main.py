"""Forecaster for make.fun roblox-up-and-coming-forecast-01.

Reads the Up-and-Coming chart history (history.ndjson) and task.json from the
input dir, and writes forecast.json: per-place probabilities for the chart's
top 10 across the scored window, plus a top-10 list.

Model, per game:
  - rank velocity from recent reads (places/hour, recency weighted),
  - playing-count growth slope (log scale) folded into the velocity,
  - damped drift + mild mean reversion for the expected rank at each future
    hour, a Gaussian place kernel whose width grows with the horizon,
  - a hard age-out: games leave the chart 28 days after creation, so their
    probability is zero for the hours after they age out,
  - the per-hour distributions are averaged over the window and each place
    is renormalized to a cap below 1, leaving an unnamed share for games
    nobody can name yet (new releases entering mid-window).
Deterministic: no randomness, no clock reads beyond the input data.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Tunable parameters (fitted on practice backtests, kept few on purpose).
P = {
    "tau_v": 30.0,        # hours; rank-velocity damping time
    "tau_r": 60.0,        # hours; mean-reversion time
    "revert_w": 0.6,      # how far to revert toward the weighted mean rank
    "slope_k": 30.0,      # playing-growth slope -> places/hour conversion
    "sigma0": 1.15,       # kernel width at horizon 0 (places)
    "sigma1": 0.5,        # kernel width growth per sqrt(hour)
    "vol_w": 0.25,        # per-game volatility widening
    "cap1": 0.96,         # named-mass cap at place 1
    "cap10": 0.84,        # named-mass cap at place 10
    "tail_mu": 26.0,      # games with expected rank beyond this get no rows
    "age_mode": "off",    # 'off' | 'soft' | 'hard': 28-day chart age-out
    "age_soft": 0.3,      # probability scale after ageing out in 'soft' mode
}


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_reads(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: parse_ts(r["at"]))
    return reads


def game_series(reads):
    """universe_id -> list of (t_hours, rank, playing); plus created map."""
    t0 = parse_ts(reads[0]["at"])
    series = {}
    created = {}
    for r in reads:
        t = (parse_ts(r["at"]) - t0) / 3600.0
        det = r.get("details") or {}
        for g in r.get("up_and_coming") or []:
            uid = g["universe_id"]
            series.setdefault(uid, []).append((t, g["rank"], g.get("playing") or 0))
            d = det.get(str(uid)) or det.get(uid)
            if d and d.get("created"):
                created[uid] = parse_ts(d["created"])
    return series, created, t0


def weighted_mean(pairs):
    tw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tw if tw else 0.0


def forecast(reads, history_until_ts, hours=72, created_override=None, params=None):
    p = dict(P)
    if params:
        p.update(params)
    series, created, t0 = game_series(reads)
    if created_override:
        for k, v in created_override.items():
            created.setdefault(int(k), v if isinstance(v, (int, float)) else parse_ts(v))
    T = (history_until_ts - t0) / 3600.0  # history end, in hours since first read

    # Per-game features.
    feats = {}
    for uid, obs in series.items():
        obs.sort()
        t_last, r_last, p_last = obs[-1]
        # Recency-weighted mean rank over the last 72h of observations.
        recent = [(r, math.exp(-(T - t) / 36.0)) for (t, r, _) in obs if T - t <= 96]
        r_mean = weighted_mean(recent) if recent else float(r_last)
        # Rank velocity: per-hour rank change between consecutive observations
        # in the last 72h, recency weighted. Positive = falling.
        vels = []
        for (t1, r1, _), (t2, r2, _) in zip(obs, obs[1:]):
            if T - t2 <= 72 and t2 > t1:
                vels.append((((r2 - r1) / (t2 - t1)), math.exp(-(T - t2) / 24.0)))
        v_rank = weighted_mean(vels) if vels else 0.0
        # Playing slope: least squares of log(playing) vs t over the last 72h.
        pts = [(t, math.log(max(pl, 1))) for (t, _, pl) in obs if T - t <= 72 and pl > 0]
        slope = 0.0
        if len(pts) >= 2 and pts[-1][0] - pts[0][0] >= 6:
            n = len(pts)
            mt = sum(x for x, _ in pts) / n
            ml = sum(y for _, y in pts) / n
            den = sum((x - mt) ** 2 for x, _ in pts)
            if den > 0:
                slope = sum((x - mt) * (y - ml) for x, y in pts) / den
            slope = max(-0.04, min(0.04, slope))
        # Rank volatility: stdev of per-interval rank changes (per sqrt hour).
        vol = 0.0
        if len(vels) >= 2:
            m = weighted_mean(vels)
            vol = math.sqrt(weighted_mean([((v - m) ** 2, w) for v, w in vels]))
        # Hours until the game ages out of the chart (28 days after creation).
        exit_h = None
        if uid in created:
            exit_h = (created[uid] + 28 * 86400 - t0) / 3600.0 - T
        feats[uid] = {
            "r_last": r_last, "r_mean": r_mean, "stale_h": T - t_last,
            "v": v_rank - slope * p["slope_k"], "vol": vol, "exit_h": exit_h,
        }

    # Only games seen on the chart recently can be forecast to hold places.
    cands = {u: f for u, f in feats.items() if f["stale_h"] <= 30 and f["r_last"] <= 60}
    if not cands:  # degenerate input: fall back to the last read's top 10
        cands = {u: f for u, f in feats.items() if f["stale_h"] <= 30}

    places = 10
    acc = {u: [0.0] * places for u in cands}   # summed per-hour probabilities
    top10_mass = {u: 0.0 for u in cands}
    for h in range(1, hours + 1):
        damp = p["tau_v"] * (1 - math.exp(-h / p["tau_v"]))
        rev = 1 - math.exp(-h / p["tau_r"])
        sigma_base = p["sigma0"] + p["sigma1"] * math.sqrt(h)
        mus = {}
        for u, f in cands.items():
            aged = f["exit_h"] is not None and h > f["exit_h"]
            if aged and p["age_mode"] == "hard":
                continue
            mu = f["r_last"] + f["v"] * damp
            mu += (f["r_mean"] - (f["r_last"] + f["v"] * damp)) * rev * p["revert_w"]
            mus[u] = (max(1.0, mu), p["age_soft"] if aged and p["age_mode"] == "soft" else 1.0)
        for u, (mu, scale) in mus.items():
            if mu > p["tail_mu"]:
                continue
            sig = sigma_base * (1 + p["vol_w"] * min(feats[u]["vol"], 3.0))
            tot = 0.0
            row = acc[u]
            for k in range(1, places + 1):
                w = math.exp(-((k - mu) ** 2) / (2 * sig * sig)) * scale
                row[k - 1] += w
                tot += w
            top10_mass[u] += tot

    rows = []
    for k in range(1, places + 1):
        cap = p["cap1"] + (p["cap10"] - p["cap1"]) * (k - 1) / (places - 1)
        col = [(u, acc[u][k - 1] / hours) for u in cands]
        s = sum(v for _, v in col)
        if s <= 0:
            continue
        scale = min(1.0, cap / s)
        for u, v in col:
            pv = v * scale
            if pv >= 0.001:
                rows.append({"universe_id": int(u), "place": k, "p": round(pv, 6)})
    rows.sort(key=lambda r: (r["place"], -r["p"]))
    top10 = [u for u, _ in sorted(top10_mass.items(), key=lambda kv: -kv[1])[:10]]
    return {"probabilities": rows, "top10": [int(u) for u in top10]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    reads = load_reads(input_dir)
    history_until = parse_ts(task["history_until"])
    window = task.get("window") or {}
    hours = int(round((parse_ts(window["until"]) - parse_ts(window["from"])) / 3600.0)) if window else 72
    fc = forecast(reads, history_until, hours=hours)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")), "w") as f:
        json.dump(fc, f)


if __name__ == "__main__":
    main()
