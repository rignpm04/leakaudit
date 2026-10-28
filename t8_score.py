# t8_score.py -- the results of fine-tune run 5, scored against test 5's answer key exactly as PREREGISTRATION.md
# fixes them. Python 3.9; press Run in PyCharm, only after (1) the pod's answers are imported (t5_import_gpu.py) and
# their fingerprints are public, and (2) the hand labels are reviewed (t7_label_review.py) and key_corrections.json is
# final (an empty list if nothing changes). No calls.
#
# Needs in this folder: test5_sealed_2026-10-26.zip (checked against its public SHA-256), key_corrections.json,
# test5_aliases.json (checked against the SHA-256 in PREREGISTRATION.md), test_prompts.jsonl, cases.json, llm_cache/,
# baselines_test5.jsonl, and the returned workbooks in labels/.
#
# Scoring (fixed in advance): a model's ratings of a table are its answers to the table's prompts (at most 10 columns
# each; a disguised prompt's names are mapped back to the real columns). A prompt with no readable answer leaves its
# columns unflagged. Debatable columns (probable) count neither way. Macro recall = the mean, over the tables with
# leaks, of the share of their leak columns flagged. Within budget = the best threshold (0-100, step 5) with at most 1
# false flag per table on average in the group (the primary setting); fixed = flagged at risk >= 50.
# Hypotheses (within budget, real test 5 unless said; trained = the mean of the three seeds' scores; primary key = the
# sealed key with the corrections of key_corrections.json):
#   H1  trained - untrained >= +0.10, and its 95% interval above 0
#   H2  best frontier model (Claude, GPT) - trained <= 0.05
#   H3  trained, real - trained, disguised <= 0.05 (the tables that have a disguise, in both conditions)
#   H4  trained - column check >= +0.20, and its 95% interval above 0
#   H5  held-out generated domains: trained - untrained >= -0.05
# Each with a 95% bootstrap interval (tables resampled 2,000 times; the threshold is tuned again in every resample).
# Output: results_run5.txt and results_run5.json.

CORRECTIONS_FILE = "key_corrections.json"
REVIEW_FILE = "label_review_run5.json"   # t7's list of the labelers' disagreements with the key
ALIASES_FILE = "test5_aliases.json"
ALIASES_SHA256 = "ca81dbb53ce50bd7d17055173a5d872d78f9979ceccee5b1d88906da8ae10615"
FRONTIER_FILES_SHA256 = "7ad2ac42ea14a068670ec7c7312356358c3cf10625dfc3561dbb09779f113fe5"   # frontier_manifest.json: the SHA-256 of its sorted file list, fixed in PREREGISTRATION.md
ANSWERS_FILES_SHA256 = "ff1c61ceb847d2bbac859425abad2c57118dad36c76a95628ec317e2919b0cb2"    # answers_manifest_run5.json: the same for the pod's answer files, entered once published
LABELS_DIR = "labels"
BASE_LABEL = "gpu5-qwen3.5-9b"
SEEDS = [3407, 3408, 3409]
SEED_LABELS = ["gpu5-leakaudit5-s%d" % s for s in SEEDS]
GGUF_LABELS = ["gguf5-q4km", "gguf5-q6k"]
FRONTIER = ["claude", "gpt"]
MOCK = False                    # True: pipeline test only (mock-claude / mock-gpt as the frontier models)
TRAIN_DOMAINS = ["claims", "orders", "saas", "shipments"]   # as in t3 / t4
HOLDOUT_DOMAINS = ["lending", "hospital"]
FIXED_T = 50
BUDGET_PER_TABLE = 1.0
BOOT = 2000
BOOT_SEED = 20261101
H1_MIN_GAIN, H2_MAX_GAP, H3_MAX_DROP, H4_MIN_MARGIN, H5_MAX_LOSS = 0.10, 0.05, 0.05, 0.20, 0.05
# the column check at the vendors' default levels (score 0-100: 100 x |2 AUC - 1| for yes/no, 100 x R^2 for numbers)
VENDOR = {"DataRobot moderate (Gini >= 0.85)": (85.0, 85.0), "DataRobot high (Gini >= 0.975)": (97.5, 97.5),
          "H2O flag (AUC >= 0.95 / R2 >= 0.95)": (90.0, 95.0), "H2O notice (AUC >= 0.80 / R2 >= 0.80)": (60.0, 80.0)}
OUT_FILE = "results_run5.txt"
OUT_JSON = "results_run5.json"

# ============================================================ internals
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import r5_common as r5  # noqa: E402
import step3_llm_audit as s3  # noqa: E402

import numpy as np  # noqa: E402

GRID = [float(t) for t in range(0, 101, 5)]
POD_FILES = {"answers_base_run5.jsonl": BASE_LABEL, "answers_gguf_q4_k_m_run5.jsonl": "gguf5-q4km",
             "answers_gguf_q6_k_run5.jsonl": "gguf5-q6k"}
POD_FILES.update({"answers_s%d_run5.jsonl" % s: "gpu5-leakaudit5-s%d" % s for s in SEEDS})
FRONT_LABELS = [("mock-" if MOCK else "") + m for m in FRONTIER]
REAL, DIS = r5.TEST5_REAL, r5.TEST5_DISGUISED
SAME = "real test 5 (tables with a disguise)"
LINES: List[str] = []
JS: Dict[str, Any] = {}


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


# ============================================================ inputs
def load_key(cols: Dict[Tuple[str, str], List[str]]
             ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """(sealed key, corrected key, the corrections, the labeling codes) for test 5: {table: {leaks, probable, kind}}.
    Every key column must be one of the table's prompt columns, and every correction must follow the sealing rule."""
    sealed = r5.read_sealed(common.path(r5.SEALED_ZIP))
    key = {c["id"]: {"leaks": set(c["leak_columns"]), "probable": set(c.get("probable_leak_columns") or []) -
                     set(c["leak_columns"]), "kind": c["kind"], "task": c["task"], "name": c["name"]}
           for c in sealed["cases"]}
    for t, k in key.items():
        absent = sorted((k["leaks"] | k["probable"]) - set(cols.get((t, "real"), [])))
        if absent:
            raise SystemExit("STOP: the key of %s names columns the prompts do not show: %s. Tell Claude." % (t, absent))
    p = common.path(CORRECTIONS_FILE)
    if not os.path.exists(p):
        raise SystemExit("%s is missing: it comes from the review of the hand labels (t7_label_review.py), before "
                         "scoring. (No corrections: Claude sends one with an empty list.)" % CORRECTIONS_FILE)
    corr = (common.load_json(p) or {}).get("corrections")
    if not isinstance(corr, list):
        raise SystemExit("%s is not a corrections file. Tell Claude." % CORRECTIONS_FILE)
    labeled = set(sealed["codes"]["codes"].values())
    fixed = {t: {"leaks": set(v["leaks"]), "probable": set(v["probable"]), "kind": v["kind"], "task": v["task"],
                 "name": v["name"]} for t, v in key.items()}
    disagreed = set()
    if corr:
        rev = common.load_json(common.path(REVIEW_FILE))
        if not isinstance(rev, dict):
            raise SystemExit("STOP: corrections need %s (t7_label_review.py) beside them. Tell Claude." % REVIEW_FILE)
        disagreed = {(d["table"], d["column"]) for d in rev.get("disagreements") or [] if not d.get("info_only")}
    seen = set()
    for c in corr:
        t, col, to = c.get("table"), c.get("column"), c.get("to")
        now = None if t not in fixed else ("leak" if col in fixed[t]["leaks"] else
                                           "probable" if col in fixed[t]["probable"] else "fine")
        if (t not in labeled or col not in cols.get((t, "real"), []) or to not in ("leak", "probable", "fine") or
                to == now or not str(c.get("evidence") or "").strip() or (t, col) not in disagreed or (t, col) in seen):
            raise SystemExit("STOP: correction %s breaks the sealing rule (a column of one of the 15 labeled tables, "
                             "exactly as the prompts name it, that a labeler disagreed on, a real change, with "
                             "evidence, listed once). Tell Claude." % c)
        seen.add((t, col))
        k = fixed[t]
        k["leaks"].discard(col)
        k["probable"].discard(col)
        if to == "leak":
            k["leaks"].add(col)
        elif to == "probable":
            k["probable"].add(col)
        k["kind"] = "leak" if k["leaks"] else "clean"
    return key, fixed, corr, sealed["codes"]


def load_prompts() -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], List[str]], Dict[str, str]]:
    rows = r5.json_lines(common.path("test_prompts.jsonl"))
    cols: Dict[Tuple[str, str], List[str]] = {}
    group: Dict[str, str] = {}
    for r in rows:
        cols.setdefault((r["table"], r["condition"]), []).extend(r["original"][c] for c in r["columns"])
        if r["condition"] == "real":
            group[r["table"]] = r["group"]
    return rows, cols, group


def check_fingerprints() -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """The test prompts, the pod's answer files and the frontier answers must be the fingerprinted ones."""
    tp = r5.sha256_file(common.path("test_prompts.jsonl"))
    if (common.load_json(common.path("test_manifest.json")) or {}).get("sha256") != tp:
        raise SystemExit("STOP: test_prompts.jsonl is not the file t4_make_test_prompts.py fingerprinted. Tell Claude.")
    am_p = common.path("answers_manifest_run5.json")
    if not os.path.exists(am_p):
        raise SystemExit("answers_manifest_run5.json is missing: run t5_import_gpu.py first.")
    am, am_sha = common.load_json(am_p) or {}, r5.sha256_file(am_p)
    if (am.get("inputs") or {}).get("test_prompts.jsonl") != tp:
        raise SystemExit("STOP: the pod answered another test_prompts.jsonl. Tell Claude.")
    a_sha = r5.canonical_sha(am.get("answers") or {})
    if (ANSWERS_FILES_SHA256 or not MOCK) and a_sha != ANSWERS_FILES_SHA256:
        raise SystemExit("STOP: the pod's answer files (%s) are not the published ones (ANSWERS_FILES_SHA256 at the "
                         "top). Tell Claude." % a_sha)
    for f, h in (am.get("answers") or {}).items():
        if not os.path.exists(common.path(f)) or r5.sha256_file(common.path(f)) != h:
            raise SystemExit("STOP: %s does not match its fingerprint. Tell Claude." % f)
    fm_p = common.path("frontier_manifest.json")
    fm = common.load_json(fm_p) if os.path.exists(fm_p) else None
    if not isinstance(fm, dict):
        raise SystemExit("frontier_manifest.json is missing (t2_teachers.py writes it). Tell Claude.")
    if (FRONTIER_FILES_SHA256 or not MOCK) and r5.canonical_sha(fm.get("files") or {}) != FRONTIER_FILES_SHA256:
        raise SystemExit("STOP: frontier_manifest.json does not list the pre-registered answer files. Tell Claude.")
    bad = [rel for rel, h in (fm.get("files") or {}).items()
           if not os.path.exists(common.path(rel)) or r5.sha256_file(common.path(rel)) != h]
    if bad:
        raise SystemExit("STOP: %d frontier answer file(s) changed since they were fingerprinted (e.g. %s). Tell Claude."
                         % (len(bad), bad[0]))
    return am, am_sha, fm


def load_answers(rows: List[Dict[str, Any]], am: Dict[str, Any], am_sha: str, fm: Dict[str, Any]
                 ) -> Tuple[Dict[str, Dict[Tuple[str, str], Dict[str, float]]], Dict[str, Dict[str, Dict[str, int]]]]:
    """{label: {(table, condition): {real column: risk}}} from llm_cache/<label>/chunks/ (answers to exactly these
    prompts only), and per label and group the prompts answered / unreadable / stale / absent. A pod answer counts only
    if it comes from the fingerprinted answer file, a frontier answer only if its file is in frontier_manifest.json."""
    answers: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}
    cover: Dict[str, Dict[str, Dict[str, int]]] = {}
    root = common.path("llm_cache")
    pod = {lab: f for f, lab in POD_FILES.items()}
    front_files = set((fm.get("files") or {}).keys())
    for lab in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        d = os.path.join(root, lab, "chunks")
        if not os.path.isdir(d):
            continue
        got: Dict[Tuple[str, str], Dict[str, float]] = {}
        cv: Dict[str, Dict[str, int]] = {}
        for r in rows:
            st = cv.setdefault(r["group"], {"answered": 0, "unreadable": 0, "stale": 0, "absent": 0})
            path = os.path.join(d, r["id"] + ".json")
            rec = common.load_json(path)
            if not isinstance(rec, dict):
                st["absent"] += 1
                continue
            if rec.get("prompt_hash") != common.text_hash(str(rec.get("model_id")), r["system"], r["user"]):
                st["stale"] += 1
                continue
            if lab in pod and (rec.get("answers_sha256") != (am.get("answers") or {}).get(pod[lab]) or
                               rec.get("manifest_sha256") != am_sha):
                st["stale"] += 1   # not from the fingerprinted file: run t5_import_gpu.py again
                continue
            if lab in FRONT_LABELS and os.path.relpath(path, HERE).replace(os.sep, "/") not in front_files:
                st["stale"] += 1   # a frontier answer that was not fingerprinted
                continue
            k = (r["table"], r["condition"])
            got.setdefault(k, {})
            if rec.get("unreadable") or not rec.get("parsed"):
                st["unreadable"] += 1
                continue
            st["answered"] += 1
            cols = rec["parsed"].get("columns") or {}
            for shown in r["columns"]:
                v = s3.to_risk((cols.get(shown) or {}).get("leak_risk"))
                if np.isfinite(v):
                    got[k][r["original"][shown]] = float(v)
        if any(x["answered"] or x["unreadable"] for x in cv.values()):
            answers[lab] = got
            cover[lab] = cv
    return answers, cover


def complete_in(cover: Dict[str, Dict[str, Dict[str, int]]], lab: str, group_name: str) -> bool:
    st = (cover.get(lab) or {}).get(group_name)
    return bool(st) and st["stale"] == 0 and st["absent"] == 0


def load_baselines() -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    p = common.path("baselines_test5.jsonl")
    if not os.path.exists(p):
        return {}
    cc: Dict[Tuple[str, str], Dict[str, float]] = {}
    kw: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in r5.json_lines(p):
        cc[(r["table"], "real")] = {c: float(v["colcheck"]) for c, v in r["columns"].items()}
        kw[(r["table"], "real")] = {c: float(v["keyword"]) for c, v in r["columns"].items()}
    return {"column check": cc, "keyword rule": kw}


# ============================================================ scoring
class Group:
    """The tables of one group with their keys, as arrays per label for fast tuning and bootstrapping."""

    def __init__(self, name: str, tables: List[str], cond: str, cols: Dict[Tuple[str, str], List[str]],
                 key: Dict[str, Dict[str, Any]], traps: Optional[Dict[str, set]] = None):
        self.name, self.tables, self.cond = name, tables, cond
        self.cols = {t: cols[(t, cond)] for t in tables}
        self.key = key
        self.traps = traps or {}
        self.has_leaks = np.array([bool(key[t]["leaks"] & set(self.cols[t])) for t in tables])

    def arrays(self, risks: Dict[Tuple[str, str], Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """(recall per table and threshold, false flags per table and threshold); tables without leaks get recall 0
        (they are left out of the mean)."""
        rec = np.zeros((len(self.tables), len(GRID)))
        ff = np.zeros((len(self.tables), len(GRID)))
        for i, t in enumerate(self.tables):
            r = risks.get((t, self.cond), {})
            leaks = self.key[t]["leaks"] & set(self.cols[t])
            skip = self.key[t]["probable"]
            v = {c: r.get(c, float("nan")) for c in self.cols[t]}
            for j, th in enumerate(GRID):
                flagged = {c for c, x in v.items() if np.isfinite(x) and x >= th - 1e-9}
                if leaks:
                    rec[i, j] = len(flagged & leaks) / float(len(leaks))
                ff[i, j] = len(flagged - leaks - skip)
        return rec, ff


def tuned_from(rec: np.ndarray, ff: np.ndarray, has: np.ndarray, idx: Optional[np.ndarray] = None
               ) -> Tuple[float, Optional[float], float]:
    """(macro recall, threshold, false flags) at the best threshold within budget, for the tables idx (all if None)."""
    if idx is None:
        idx = np.arange(rec.shape[0])
    h = has[idx]
    if not h.any():
        return float("nan"), None, 0.0
    macro = rec[idx][h].mean(axis=0)
    ffs = ff[idx].sum(axis=0)
    budget = BUDGET_PER_TABLE * len(idx)
    best = None
    for j in range(len(GRID)):
        if ffs[j] <= budget + 1e-9:
            k = (round(float(macro[j]), 6), -float(ffs[j]))
            if best is None or k > best[0]:
                best = (k, j)
    if best is None:
        return 0.0, None, 0.0
    j = best[1]
    return float(macro[j]), GRID[j], float(ffs[j])


def fixed_from(rec: np.ndarray, ff: np.ndarray, has: np.ndarray, t: float = FIXED_T) -> Tuple[float, float]:
    j = GRID.index(t)
    return (float(rec[has, j].mean()) if has.any() else float("nan")), float(ff[:, j].sum())


def flags_at(g: Group, risks: Dict[Tuple[str, str], Dict[str, float]], thresholds: Dict[str, float]
             ) -> Tuple[float, int, int, int]:
    """Macro recall, false flags, leaks found and leaks, flagging at a per-task threshold (vendor defaults)."""
    rec, ff, found, n = [], 0, 0, 0
    for t in g.tables:
        r = risks.get((t, g.cond), {})
        th = thresholds["binary" if g.key[t].get("task") == "binary" else "regression"]
        leaks = g.key[t]["leaks"] & set(g.cols[t])
        flagged = {c for c in g.cols[t] if np.isfinite(r.get(c, float("nan"))) and r[c] >= th - 1e-9}
        ff += len(flagged - leaks - g.key[t]["probable"])
        if leaks:
            rec.append(len(flagged & leaks) / float(len(leaks)))
            found += len(flagged & leaks)
            n += len(leaks)
    return (float(np.mean(rec)) if rec else float("nan")), ff, found, n


def ci(vals: List[float]) -> Tuple[float, float]:
    v = np.array([x for x in vals if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"), float("nan"))


# ============================================================ main
def main() -> None:
    rows, cols, group = load_prompts()
    am, am_sha, front_man = check_fingerprints()
    sealed_key, fixed_key, corrections, codes = load_key(cols)
    aliases_p = common.path(ALIASES_FILE)
    if not os.path.exists(aliases_p) or r5.sha256_file(aliases_p) != ALIASES_SHA256:
        raise SystemExit("STOP: %s is missing or not the pre-registered file (SHA-256). Tell Claude." % ALIASES_FILE)
    aliases = common.load_json(aliases_p)["tables"]
    gen = {c["id"]: c for c in common.load_cases(quiet=True) if c.get("url") == "generated"}
    gen_key = {t: {"leaks": set(c["leak_columns"]), "probable": set(c.get("probable_leak_columns") or []),
                   "kind": c["kind"], "task": c["task"]} for t, c in gen.items() if t in group}
    traps = {t: set((c.get("trap_columns") or {}).keys()) for t, c in gen.items() if t in group}
    for t, k in gen_key.items():
        absent = sorted((k["leaks"] | k["probable"] | traps[t]) - set(cols.get((t, "real"), [])))
        if absent:
            raise SystemExit("STOP: generated table %s's key names columns the prompts do not show: %s." % (t, absent))
    answers, cover = load_answers(rows, am, am_sha, front_man)
    answers.update(load_baselines())
    front = FRONT_LABELS
    t5 = sorted(t for t, g in group.items() if g == REAL)
    dis_tables = sorted({t for (t, c) in cols if c == "disguised"})
    if len(t5) != 68 or set(t5) != set(sealed_key):
        raise SystemExit("STOP: the test prompts do not cover exactly the 68 sealed tables. Tell Claude.")
    need = [(lab, g) for lab in [BASE_LABEL] + SEED_LABELS for g in r5.GROUPS] + \
        [(lab, g) for lab in front for g in (REAL, DIS)]
    gaps = ["%s / %s: %s" % (lab, g, (cover.get(lab) or {}).get(g)) for lab, g in need if not complete_in(cover, lab, g)]
    if gaps:
        raise SystemExit("STOP: answers are missing or do not match the prompts for: %s. Run t5_import_gpu.py (pod "
                         "answers) or t2_teachers.py (Claude, GPT) until complete, or tell Claude." % "; ".join(gaps[:8]))
    if "column check" in answers and set(t for (t, _) in answers["column check"]) != set(t5):
        raise SystemExit("STOP: baselines_test5.jsonl does not cover the 68 tables: run t6_baselines.py again.")
    out("Fine-tune run 5 -- results against test 5's key (%s)%s" % (common.stamp(), " [MOCK]" if MOCK else ""))
    out("Key corrections applied (primary key): %d%s" % (len(corrections), "".join(
        "\n  %s / %s -> %s (%s)" % (c["table"], c["column"], c["to"], " ".join(str(c.get("why", "")).split())[:160])
        for c in corrections)))
    JS["corrections"] = corrections
    results: Dict[str, Any] = {}
    for kname, key in (("primary", fixed_key), ("sealed", sealed_key)):
        allkey = dict(gen_key)
        allkey.update(key)
        groups = {g: Group(g, sorted(t for t, gg in group.items() if gg == g), "real", cols, allkey, traps)
                  for g in r5.GEN_GROUPS}
        groups[REAL] = Group(REAL, t5, "real", cols, allkey)
        groups[SAME] = Group(SAME, dis_tables, "real", cols, allkey)
        groups[DIS] = Group(DIS, dis_tables, "disguised", cols, allkey)
        arr = {lab: {g: groups[g].arrays(answers[lab]) for g in groups} for lab in answers}
        res: Dict[str, Dict[str, Any]] = {}
        for lab in arr:
            res[lab] = {}
            for g, G in groups.items():
                if not any((t, G.cond) in answers[lab] for t in G.tables):
                    continue
                rec, ff = arr[lab][g]
                m, th, f = tuned_from(rec, ff, G.has_leaks)
                fm, ffm = fixed_from(rec, ff, G.has_leaks)
                res[lab][g] = {"budget": m, "threshold": th, "ff": f, "fixed": fm, "fixed_ff": ffm, "tables": len(G.tables)}
        missing = [x for x in [BASE_LABEL] + SEED_LABELS if x not in res]
        if missing:
            raise SystemExit("No answers for %s: run t5_import_gpu.py first." % ", ".join(missing))
        trained = {g: float(np.mean([res[s][g]["budget"] for s in SEED_LABELS])) for g in groups}
        best_front = max((f for f in front if f in res and REAL in res[f]), key=lambda f: res[f][REAL]["budget"],
                         default=None)

        # bootstrap: tables resampled within each group (the disguise pair on the same tables)
        rng = np.random.RandomState(BOOT_SEED)
        boot = {h: [] for h in ("H1", "H2", "H3", "H4", "H5")}
        nR, nS, nH = len(groups[REAL].tables), len(groups[SAME].tables), len(groups["held-out domains"].tables)
        for _ in range(BOOT):
            iR, iS, iH = rng.randint(0, nR, nR), rng.randint(0, nS, nS), rng.randint(0, nH, nH)

            def tb(lab: str, g: str, idx: np.ndarray) -> float:
                rec, ff = arr[lab][g]
                return tuned_from(rec, ff, groups[g].has_leaks, idx)[0]
            tr_R = np.mean([tb(s, REAL, iR) for s in SEED_LABELS])
            boot["H1"].append(tr_R - tb(BASE_LABEL, REAL, iR))
            if best_front:
                boot["H2"].append(max(tb(f, REAL, iR) for f in front if f in arr) - tr_R)
            boot["H3"].append(np.mean([tb(s, SAME, iS) for s in SEED_LABELS]) -
                              np.mean([tb(s, DIS, iS) for s in SEED_LABELS]))
            if "column check" in arr:
                boot["H4"].append(tr_R - tb("column check", REAL, iR))
            boot["H5"].append(np.mean([tb(s, "held-out domains", iH) for s in SEED_LABELS]) -
                              tb(BASE_LABEL, "held-out domains", iH))
        est = {"H1": trained[REAL] - res[BASE_LABEL][REAL]["budget"],
               "H2": (max(res[f][REAL]["budget"] for f in front if f in res) - trained[REAL]) if best_front else float("nan"),
               "H3": trained[SAME] - trained[DIS],
               "H4": (trained[REAL] - res["column check"][REAL]["budget"]) if "column check" in res else float("nan"),
               "H5": trained["held-out domains"] - res[BASE_LABEL]["held-out domains"]["budget"]}
        rule = {"H1": ("trained - untrained, real test 5", ">=", H1_MIN_GAIN),
                "H2": ("best frontier - trained, real test 5", "<=", H2_MAX_GAP),
                "H3": ("trained real - trained disguised (same tables)", "<=", H3_MAX_DROP),
                "H4": ("trained - column check, real test 5", ">=", H4_MIN_MARGIN),
                "H5": ("trained - untrained, held-out domains", ">=", -H5_MAX_LOSS)}
        hyp = {}
        for h, (what, op, lim) in rule.items():
            e = est[h]
            ok = (e >= lim - 1e-9) if op == ">=" else (e <= lim + 1e-9)
            lo, hi = ci(boot[h])
            if h in ("H1", "H4"):   # the two claims of superiority also need the interval above zero
                ok = ok and np.isfinite(lo) and lo > 0
            hyp[h] = {"what": what, "estimate": e, "rule": "%s %+.2f%s" % (op, lim, ", interval > 0" if h in ("H1", "H4")
                                                                         else ""),
                      "supported": bool(np.isfinite(e) and ok), "ci95": [lo, hi]}
        results[kname] = {"labels": res, "trained_mean": trained, "hypotheses": hyp, "best_frontier": best_front,
                          "groups": {g: len(G.tables) for g, G in groups.items()}}
        if kname == "primary":
            prim_groups = groups

    # ---------------------------------------------------------------- report
    P = results["primary"]
    out("")
    out("HYPOTHESES (within budget, <= %.0f false flag per table; trained = the mean of seeds %s; primary key)"
        % (BUDGET_PER_TABLE, ", ".join(map(str, SEEDS))))
    rows_h = []
    for h, d in P["hypotheses"].items():
        s = results["sealed"]["hypotheses"][h]
        rows_h.append([h, d["what"], "%+.3f" % d["estimate"], "[%+.3f, %+.3f]" % tuple(d["ci95"]), d["rule"],
                       "SUPPORTED" if d["supported"] else "not supported",
                       "%+.3f %s" % (s["estimate"], "supported" if s["supported"] else "not supported")])
    out(common.format_table(["", "comparison", "estimate", "95% interval", "rule", "verdict", "on the sealed key"],
                            rows_h))
    out("")
    order = [BASE_LABEL] + SEED_LABELS + [x for x in GGUF_LABELS if x in P["labels"]] + \
        [x for x in front if x in P["labels"]] + [x for x in ("column check", "keyword rule") if x in P["labels"]]
    show_groups = [REAL, DIS, SAME] + r5.GEN_GROUPS
    for setting in ("budget", "fixed"):
        out("WITHIN BUDGET (best threshold, <= 1 false flag per table)" if setting == "budget" else
            "FIXED (flagged at risk >= %d)" % FIXED_T)
        tab = []
        for lab in order:
            row = [lab]
            for g in show_groups:
                d = P["labels"][lab].get(g)
                if not d:
                    row.append("-")
                elif setting == "budget":
                    row.append("%.3f / %.0f ff (T %s)" % (d["budget"], d["ff"], "-" if d["threshold"] is None
                                                          else "%.0f" % d["threshold"]))
                else:
                    row.append("%.3f / %.0f ff" % (d["fixed"], d["fixed_ff"]))
            tab.append(row)
        tab.append(["trained (mean of seeds)"] + ["%.3f" % P["trained_mean"][g] if setting == "budget" else
                                                  "%.3f" % float(np.mean([P["labels"][s][g]["fixed"] for s in SEED_LABELS]))
                                                  for g in show_groups])
        out(common.format_table(["model"] + ["%s (%d)" % (g, P["groups"][g]) for g in show_groups], tab))
        out("")
    spread = [P["labels"][s][REAL]["budget"] for s in SEED_LABELS]
    out("Seed spread on real test 5 (within budget): %s; range %.3f" % (", ".join("%.3f" % x for x in spread),
                                                                       max(spread) - min(spread)))
    G = prim_groups["trap tables"]
    tab = []
    for lab in [x for x in order if x in answers and any((t, "real") in answers[x] for t in G.tables)]:
        lh = ln = th = tn = other = 0
        for t in G.tables:
            r = answers[lab].get((t, "real"), {})
            flagged = {c for c in G.cols[t] if np.isfinite(r.get(c, float("nan"))) and r[c] >= FIXED_T - 1e-9}
            leaks = G.key[t]["leaks"] & set(G.cols[t])
            lh, ln = lh + len(flagged & leaks), ln + len(leaks)
            th, tn = th + len(flagged & traps[t]), tn + len(traps[t])
            other += len(flagged - leaks - traps[t] - G.key[t]["probable"])
        tab.append([lab, "%d / %d" % (lh, ln), "%d / %d" % (th, tn), other])
    out("TRAP TABLES (fixed at %d, as in run 4): leaks found, trap columns flagged (legitimate columns built to look "
        "late), other false flags" % FIXED_T)
    out(common.format_table(["model", "leaks found", "trap columns flagged", "other false flags"], tab))

    # the column check at the vendors' default levels
    if "column check" in answers:
        G = prim_groups[REAL]
        out("")
        out("COLUMN CHECK AT THE VENDORS' DEFAULT LEVELS (real test 5): macro / false flags / leaks found")
        tab = []
        for nm, (b, r) in VENDOR.items():
            m, ff, found, n = flags_at(G, answers["column check"], {"binary": b, "regression": r})
            tab.append([nm, "%.3f" % m, ff, "%d / %d" % (found, n)])
        out(common.format_table(["level", "macro", "false flags", "leaks found"], tab))
        weak = [(t, c) for t in G.tables for c in fixed_key[t]["leaks"] & set(G.cols[t])
                if answers["column check"].get((t, "real"), {}).get(c, 0.0) <
                (60.0 if fixed_key[t]["task"] == "binary" else 80.0)]
        caught = {s: sum(1 for t, c in weak if answers[s].get((t, "real"), {}).get(c, -1) >= FIXED_T)
                  for s in SEED_LABELS}
        out("Leak columns below H2O's notice level: %d of %d; flagged at %d by the trained runs: %s"
            % (len(weak), sum(len(fixed_key[t]["leaks"] & set(G.cols[t])) for t in G.tables), FIXED_T,
               ", ".join("%s %d" % kv for kv in caught.items())))

    # hand labels on the 15 tables
    out("")
    files = sorted(f for f in glob.glob(os.path.join(common.path(LABELS_DIR), "*.xlsx"))
                   if not os.path.basename(f).startswith("~$"))
    by_code = codes["codes"]
    lab15 = sorted(by_code.values())
    G15 = Group("15 labeled tables", lab15, "real", cols, dict(fixed_key))
    G15s = Group("15 labeled tables, sealed key", lab15, "real", cols, dict(sealed_key))
    human: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {}
    view_bad = {}
    prompt_lines: Dict[Tuple[str, str], str] = {}   # (table, column) -> the examples the model saw
    for r in rows:
        if r["condition"] == "real" and r["table"] in lab15:
            for line in r["user"].split("\n"):
                for c in r["columns"]:
                    head = "- %s | " % s3.shown_name(c)
                    if line.startswith(head) and (r["table"], c) not in prompt_lines:
                        prompt_lines[(r["table"], c)] = line
    for f in files:
        cells = r5.read_xlsx(f)
        name = str((cells.get("Start here") or {}).get("B3") or "").strip() or os.path.basename(f)
        got: Dict[Tuple[str, str], Dict[str, float]] = {}
        bad, ties = 0, 0
        for code, t in by_code.items():
            sh = cells.get(code) or {}
            got[(t, "real")] = {}
            k = 9
            while sh.get("A%d" % k) is not None:
                col = str(sh.get("B%d" % k))
                head = "- %s | %s | %s missing | %s distinct | " % (s3.shown_name(col), sh.get("C%d" % k),
                                                                    sh.get("D%d" % k), sh.get("E%d" % k))
                seen = prompt_lines.get((t, col), "")
                if seen != head + str(sh.get("F%d" % k)):
                    if seen.startswith(head) and r5.ties_only(seen[len(head):], sh.get("F%d" % k)):
                        ties += 1
                    else:
                        bad += 1
                lab = str(sh.get("G%d" % k) or "").strip().lower()
                if lab in ("leak", "fine"):
                    got[(t, "real")][col] = 100.0 if lab == "leak" else 0.0
                k += 1
        who = "human: %s (%s)" % (name, os.path.basename(f))   # by file: two workbooks never share a row
        human[who] = got
        view_bad[who] = "%d differ, %d only in tied categories" % (bad, ties)
    if human:
        out("HAND LABELS ON THE 15 LABELED TABLES (fixed: 'leak' = flagged) and every model on the same tables "
            "(fixed at %d / within budget on these 15), on the primary and on the sealed key" % FIXED_T)
        tab = []
        for lab, risks in list(human.items()) + [(x, answers[x]) for x in order if x in answers]:
            row = [lab]
            for G in (G15, G15s):
                rec, ff = G.arrays(risks)
                fm_, ffm = fixed_from(rec, ff, G.has_leaks)
                m, th, f_ = tuned_from(rec, ff, G.has_leaks)
                row += ["%.3f / %.0f ff" % (fm_, ffm), "-" if lab.startswith("human") else "%.3f / %.0f ff" % (m, f_)]
            tab.append(row)
        out(common.format_table(["labeler or model", "fixed", "within budget", "fixed (sealed key)",
                                 "within budget (sealed key)"], tab))
        out("Workbook rows against the model's view: %s" % "; ".join("%s: %s" % kv for kv in view_bad.items()))
    else:
        out("No returned workbooks in %s/: hand labels not scored." % LABELS_DIR)

    # recognition (frontier probes) and the sentences-only check
    out("")
    out("RECOGNITION (the frontier models' top-3 guesses of the dataset, test 5): tables whose source / edition was named")
    tab = []
    for m in FRONTIER:
        lab = ("mock-" if MOCK else "") + m
        row = [lab]
        for cond in ("real", "disguised"):
            src = ed = n = 0
            for t in t5:
                rel = "llm_cache/%s/probe_%s/%s.json" % (lab, cond, t)
                rec = common.load_json(common.path(rel)) if rel in front_man.get("files", {}) else None
                if not isinstance(rec, dict) or not isinstance((rec.get("parsed") or {}).get("guesses"), list):
                    continue
                n += 1
                gs = [g.get("name") for g in rec["parsed"]["guesses"]]
                src += any(r5.names_source(g, aliases[t]) for g in gs)
                ed += any(r5.names_edition(g, aliases[t]) for g in gs)
            row.append("%d / %d / %d" % (src, ed, n))
        tab.append(row)
    out(common.format_table(["model", "real: source / edition / asked", "disguised: source / edition / asked"], tab))
    out("SENTENCES-ONLY CHECK (from the three task sentences alone, the likelihood of a leak; AUC leak vs clean tables, "
        "0.5 = no information)")
    tab = []
    for m in FRONTIER:
        lab = ("mock-" if MOCK else "") + m
        pos, neg = [], []
        for t in t5:
            rel = "llm_cache/%s/sentences/%s.json" % (lab, t)
            rec = common.load_json(common.path(rel)) if rel in front_man.get("files", {}) else None
            v = ((rec or {}).get("parsed") or {}).get("likelihood")
            if v is None:
                continue
            (pos if fixed_key[t]["leaks"] else neg).append(float(v))
        tab.append([lab, len(pos), len(neg), "%.3f" % r5.auc(pos, neg)])
    out(common.format_table(["model", "leak tables", "clean tables", "AUC"], tab))

    # coverage
    out("")
    def total(lab: str, k: str) -> int:
        return sum(x[k] for x in cover[lab].values())
    out("Answers used (prompts answered / unreadable / not matching or not fingerprinted / absent): " + "; ".join(
        "%s %d/%d/%d/%d" % (lab, total(lab, "answered"), total(lab, "unreadable"), total(lab, "stale"),
                            total(lab, "absent")) for lab in sorted(cover) if lab in order))
    JS["coverage"] = {lab: cover[lab] for lab in cover if lab in order}
    JS.update(results=results, created=common.stamp(), aliases_sha256=ALIASES_SHA256, sealed_sha256=r5.SEALED_SHA256)
    with open(common.path(OUT_FILE), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")

    def plain(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        if isinstance(x, set):
            return sorted(x)
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, (np.floating, np.integer)):
            return plain(float(x))
        return x
    common.write_json(common.path(OUT_JSON), plain(JS))
    print("\n-> %s and %s. Send both to Claude." % (OUT_FILE, OUT_JSON))


if __name__ == "__main__":
    main()
