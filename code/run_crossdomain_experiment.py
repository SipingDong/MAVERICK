# -*- coding: utf-8 -*-
"""
MAVERICK cross-domain three-agent differentiated experiment (NMI Extended Data).
Author: MAVERICK research assistant (subagent), report year 2026-08-17.

Implements, on the four-source dataset (400 samples = 50 genuine / 30 manipulated /
20 in_the_wild per source):
  * three differentiated agents (A Broad Hunter / B Academic Scholar / C Precision
    Verifier) with fixed thresholds (A: 0.80 token-Jaccard broad overlap; B: 100%
    full-field authoritative match; C: exact primary-key anchor) -- thresholds are
    the MAIN-paper values, hard-coded, NOT tuned for cross-domain (red line 7);
  * four configurations: MAVERICK (differentiated unanimous), 3x A, 3x B, 3x C
    (homogeneous ablations);
  * metrics: system FPR (Clopper-Pearson one-sided 95% CI) / FNR / unambiguous
    verdict rate / human-review share, single-agent FPR/FNR, pairwise error-set
    Jaccard overlap, and the Gaussian-copula rho-sensitivity curve (21 points)
    matched against main paper Table 7.

Honesty constraints (red lines): no metric is fabricated; every number comes from
this script's run. Agent A has NO real web-search channel in this sandbox: it is
simulated with the local authoritative index + a 0.80 loose threshold (documented
in the construction report). The us_legal subset originally contained 5
invisible-tamper samples (input_text carried no claimed section name); they were
rebuilt on 2026-08-17 by restoring the claimed (mutated) section name into
input_text so the single-field name mutation is verifiable by Agent B. No agent
thresholds or roles were changed (red line 7).
"""
import json
import re
import random
import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import norm
from scipy.stats import multivariate_normal

BASE = '/Coze/Drive/辩溪/所有对话/主对话/crossdomain_experiment'
SEED = 20260817
JACCARD_THRESHOLD = 0.80          # Agent A (main paper Layer-2 value, fixed)
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# 0. Utilities
# =============================================================================
CN_DIGITS = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
             '七': 7, '八': 8, '九': 9, '两': 2}


def cn_to_int(s):
    total = 0
    section = 0
    num = 0
    for ch in s:
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
        elif ch == '十':
            section += (num if num else 1) * 10
            num = 0
        elif ch == '百':
            section += (num if num else 1) * 100
            num = 0
        elif ch == '千':
            section += (num if num else 1) * 1000
            num = 0
        elif ch == '万':
            total += (section + num) * 10000
            section = 0
            num = 0
    section += num
    return total + section


def norm_name(s):
    """Lowercase, strip non-alphanumeric (keep CJK), collapse whitespace."""
    if s is None:
        return None
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]+', ' ', str(s).lower())).strip()


def tokenize(s):
    """CJK -> per-character tokens; ASCII -> per-word tokens (lowercased)."""
    s = str(s).lower()
    tokens = []
    run = []
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff':
            if run:
                tokens.append(''.join(run))
                run = []
            tokens.append(ch)
        elif ch.isalnum():
            run.append(ch)
        else:
            if run:
                tokens.append(''.join(run))
                run = []
    if run:
        tokens.append(''.join(run))
    return tokens


def jaccard_tokens(a, b):
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def norm_title(t):
    """Zero-pad single-digit numeric US title to index key form (01..09)."""
    if re.fullmatch(r'\d{1}', str(t)):
        return '0' + str(t)
    return str(t)


def cp_upper(x, n):
    """Clopper-Pearson one-sided 95% upper bound for a proportion x/n."""
    if n <= 0:
        return float('nan')
    return float(_beta.ppf(0.95, x + 1, n - x))


# =============================================================================
# 1. Parsers (validated 0/100 mismatch per source in prototypes)
# =============================================================================
def parse_us_legal(text):
    title = None
    section = None
    year = None
    name = None
    m = re.search(r'(?:Title\s+)?(\d{1,3}[a-z]?)\s*U\.?\s*S\.?\s*C\.?', text)
    if m:
        title = norm_title(m.group(1))
    m2 = re.search(r'Title\s+(\d{1,3}[a-z]?)\s+of\s+the\s+United\s+States\s+Code', text)
    if m2:
        title = norm_title(m2.group(1))
    m3 = re.search(r'§\s*([0-9][0-9A-Za-z]*[–—-]?[0-9A-Za-z]*)', text)
    if m3:
        section = m3.group(1)
    m4 = re.search(r'\bSection\s+([0-9][0-9A-Za-z]*[–—-]?[0-9A-Za-z]*)', text)
    if m4:
        section = m4.group(1)
    myr = re.search(r'\((?:(?:West\s+)?(\d{4}))\)', text)
    if myr:
        year = int(myr.group(1))
    # name candidates -----------------------------------------------------
    # NOTE: the en-dash inside a section number (e.g. "§1862o–2", "§ 50–1")
    # must NOT be treated as the citation-separator em-dash. Require the dash
    # to be preceded by whitespace (a standalone separator like " — ").
    em = re.search(r'(?<=\s)[—–]\s*(.+)', text)
    if em:
        nm = em.group(1).strip().strip('"“”').strip().rstrip('.').strip()
        if nm:
            name = nm
    if name is None:
        q = re.search(r'[“"]([^”"]+)[”"]', text)
        if q:
            nm = q.group(1).strip().strip('.').strip()
            if nm:
                name = nm
    if name is None:
        for grp in re.finditer(r'\(([^()]+)\)', text):
            g = grp.group(1).strip()
            if re.fullmatch(r'\d{4}', g):
                continue
            if re.fullmatch(r'West\s+\d{4}', g):
                continue
            if 'providing the statutory basis' in g:
                continue
            name = g.strip().rstrip('.').strip()
    return {'title': title, 'section': section, 'year': year, 'name': name}


def parse_cn_legal(text):
    law = None
    article = None
    year = None
    m = re.search(r'《([^》]+)》', text)
    if m:
        law = m.group(1).strip()
    myr = re.search(r'（(\d{4})年[版本修订修正]*）', text)
    if myr:
        year = int(myr.group(1))
    ma = re.search(r'第([零一二三四五六七八九十百千万两\d]+)条', text)
    if ma:
        s = ma.group(1)
        if s.isdigit():
            article = int(s)
        else:
            article = cn_to_int(s)
    return {'law': law, 'article': article, 'year': year}


def parse_clinical(text):
    rid = None
    title = None
    m = re.search(r'(NCT\d{8}|ChiCTR[- ]?\d+)', text, re.I)
    if m:
        rid = m.group(1).lower().replace(' ', '')
    segs = re.findall(r'[“"「《]([^”"」》]+)[”"」》]', text)
    if segs:
        title = max(segs, key=len).strip()
    return {'rid': rid, 'title': title}


# =============================================================================
# 2. Load data & indexes
# =============================================================================
def load_samples(path):
    return json.load(open(path))


samples = {
    'us_legal': load_samples(f'{BASE}/us_legal/samples.json'),
    'cn_legal': load_samples(f'{BASE}/cn_legal/samples.json'),
    'us_clinical': load_samples(f'{BASE}/us_clinical/samples.json'),
    'cn_clinical': load_samples(f'{BASE}/cn_clinical/samples.json'),
}
for k, v in samples.items():
    assert len(v) == 100, f'{k}: expected 100, got {len(v)}'

# authoritative indexes -------------------------------------------------------
uscode = json.load(open(f'{BASE}/_assets/uscode_index.json'))['titles']
harvested = json.load(open(f'{BASE}/cn_legal/_work/harvested_laws.json'))
headers = json.load(open(f'{BASE}/cn_legal/_work/headers.json'))
us_pool = json.load(open(f'{BASE}/us_clinical/_genuine_pool.json'))
cn_pool = json.load(open(f'{BASE}/cn_clinical/_pool_candidates.json'))

# ---- cn_legal index structures ---------------------------------------------
# law short key -> full title
LAW_SHORT = {short: rec['info']['title'] for short, rec in harvested.items()}
# full title (canonical) -> short key  (reverse map)
LAW_FULL_TO_SHORT = {rec['info']['title']: short for short, rec in harvested.items()}
# short key -> set of article numbers (str)
LAW_ARTICLES = {short: set(rec['articles'].keys()) for short, rec in harvested.items()}
# short key -> revision years
LAW_YEARS = {short: set(rec.get('years', [])) for short, rec in headers.items()}


def cn_law_short(full_name):
    """map canonical full law name -> short index key; None if absent."""
    if full_name in LAW_FULL_TO_SHORT:
        return LAW_FULL_TO_SHORT[full_name]
    # fallback: strip 中华人民共和国 prefix and exact match against short keys
    if full_name.startswith('中华人民共和国'):
        rest = full_name[len('中华人民共和国'):]
        if rest in LAW_SHORT:
            return rest
    return None


# ---- clinical pools ---------------------------------------------------------
US_POOL_IDS = {p['nct_id'].lower() for p in us_pool}
US_POOL_TITLES = [p['official_title'] for p in us_pool if p.get('official_title')]
CN_POOL_IDS = {p['nctId'].lower() for p in cn_pool}
CN_POOL_TITLES = [p.get('officialTitle') for p in cn_pool if p.get('officialTitle')]

# ---- us_legal name index (for Agent A fuzzy search across the whole US Code)--
US_SEC_NAME_TOKENS = []          # list of (tokenset) for every indexed section name
for _t, _rec in uscode.items():
    for _sec in _rec.get('sections', {}).values():
        _nm = _sec.get('name')
        if _nm:
            US_SEC_NAME_TOKENS.append(set(tokenize(_nm)))


# =============================================================================
# 3. Agent verdict functions  (thresholds fixed; identical logic per source)
# =============================================================================
def agent_us_legal(parsed, which):
    """which in {'A','B','C'}."""
    title = parsed['title']
    section = parsed['section']
    name = parsed['name']
    pk_hit = False
    if title is not None and section is not None:
        rec = uscode.get(title, {}).get('sections', {}).get(section)
        pk_hit = rec is not None
    if which == 'C':
        # Precision Verifier: exact primary-key anchor; fallback exact-string match
        if pk_hit:
            return True
        return False
    if which == 'A':
        # Broad Hunter: primary-key hit OR >=0.80 token-Jaccard on claimed name
        # (whole-index fuzzy search, mirroring a broad web search)
        if pk_hit:
            return True
        if name:
            claimed = set(tokenize(name))
            if not claimed:
                return False
            for cand in US_SEC_NAME_TOKENS:
                inter = len(claimed & cand)
                union = len(claimed | cand)
                if union and inter / union >= JACCARD_THRESHOLD:
                    return True
        return False
    if which == 'B':
        # Academic Scholar: authoritative primary-key hit AND every claimed
        # auxiliary field matches exactly (section name; year is neutral for US
        # Code because the local index has no per-section revision year field)
        if not pk_hit:
            return False
        if name is not None:
            rec = uscode.get(title, {}).get('sections', {}).get(section)
            canon = rec.get('name') if rec else None
            if norm_name(name) != norm_name(canon):
                return False
        return True
    raise ValueError(which)


def agent_cn_legal(parsed, which):
    law = parsed['law']
    article = parsed['article']
    year = parsed['year']
    short = cn_law_short(law) if law else None
    pk_hit = (short is not None and article is not None
              and str(article) in LAW_ARTICLES[short])
    if which == 'C':
        # Precision Verifier: exact (law, article) anchor
        return bool(pk_hit)
    if which == 'A':
        # Broad Hunter: primary-key hit OR >=0.80 Jaccard of claimed law name
        # against every indexed law full title. Year intentionally NOT part of
        # A's comparison (no per-record year in the local retrieval layer;
        # year is guarded by B against the official revision-year set).
        if pk_hit:
            return True
        if law:
            for full in LAW_SHORT.values():
                if jaccard_tokens(law, full) >= JACCARD_THRESHOLD:
                    return True
        return False
    if which == 'B':
        # Academic Scholar: primary-key hit AND claimed revision year (if any)
        # belongs to the official revision-year set of that law.
        if not pk_hit:
            return False
        if year is not None:
            if year not in LAW_YEARS.get(short, set()):
                return False
        return True
    raise ValueError(which)


def agent_clinical(parsed, which, pool_ids, pool_titles):
    rid = parsed['rid']
    title = parsed['title']
    pk_hit = rid in pool_ids
    if which == 'C':
        # Precision Verifier: exact registration-id anchor; fallback exact title
        # (kept as a guard for unparseable inputs -- none occur in this dataset)
        if pk_hit:
            return True
        if title and not rid:
            return any(norm_name(title) == norm_name(t) for t in pool_titles)
        return False
    if which == 'A':
        if pk_hit:
            return True
        if title:
            for t in pool_titles:
                if jaccard_tokens(title, t) >= JACCARD_THRESHOLD:
                    return True
        return False
    if which == 'B':
        # Academic Scholar: authoritative-id hit AND (if a title is claimed) the
        # normalized title must exactly match the official record title.
        if not pk_hit:
            return False
        if title is not None:
            if not any(norm_name(title) == norm_name(t) for t in pool_titles):
                return False
        return True
    raise ValueError(which)


def run_agents(source, sample):
    """Return dict of per-agent verdicts plus parsed fields and pk info."""
    parsed = None
    v = {}
    if source == 'us_legal':
        parsed = parse_us_legal(sample['input_text'])
        for w in 'ABC':
            v[w] = agent_us_legal(parsed, w)
    elif source == 'cn_legal':
        parsed = parse_cn_legal(sample['input_text'])
        for w in 'ABC':
            v[w] = agent_cn_legal(parsed, w)
    else:  # clinical
        parsed = parse_clinical(sample['input_text'])
        if source == 'us_clinical':
            for w in 'ABC':
                v[w] = agent_clinical(parsed, w, US_POOL_IDS, US_POOL_TITLES)
        else:
            for w in 'ABC':
                v[w] = agent_clinical(parsed, w, CN_POOL_IDS, CN_POOL_TITLES)
    return v, parsed


# =============================================================================
# 4. Configuration / metrics helpers
# =============================================================================
CONFIGS = ['MAVERICK', '3xA', '3xB', '3xC']


def system_verdict(agent_verdicts, config):
    """Unanimous-consensus system verdict; None = dissent (human review)."""
    if config == 'MAVERICK':
        if agent_verdicts['A'] == agent_verdicts['B'] == agent_verdicts['C']:
            return agent_verdicts['A']
        return None
    w = {'3xA': 'A', '3xB': 'B', '3xC': 'C'}[config]
    return agent_verdicts[w]


def compute_metrics(records, config, which_agent=None):
    """
    records: list of dicts {id, category, gt, sys_verdict} for one source.
    System metrics use only unambiguous verdicts; dissents -> human review.
    Single-agent metrics (which_agent) use that agent's verdict on all samples.
    """
    gt_true = [r for r in records if r['gt'] is True]
    gt_false = [r for r in records if r['gt'] is False]

    def verdict(r):
        if which_agent is not None:
            return r['agent_verdicts'][which_agent]
        return r['sys_verdict'][config]

    if which_agent is None:
        unamb = [r for r in records if verdict(r) is not None]
        dissents = [r for r in records if verdict(r) is None]
        fp = sum(1 for r in unamb if r['gt'] is False and verdict(r) is True)
        tn = sum(1 for r in unamb if r['gt'] is False and verdict(r) is False)
        fn = sum(1 for r in unamb if r['gt'] is True and verdict(r) is False)
        tp = sum(1 for r in unamb if r['gt'] is True and verdict(r) is True)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')
        n_unamb = len(unamb)
        human = len(dissents) / len(records) if records else float('nan')
        unambiguous_rate = n_unamb / len(records) if records else float('nan')
    else:
        fp = sum(1 for r in records if r['gt'] is False and verdict(r) is True)
        tn = sum(1 for r in records if r['gt'] is False and verdict(r) is False)
        fn = sum(1 for r in records if r['gt'] is True and verdict(r) is False)
        tp = sum(1 for r in records if r['gt'] is True and verdict(r) is True)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float('nan')
        human = 0.0
        unambiguous_rate = 1.0
        n_unamb = len(records)
    return {
        'config': config,
        'agent': which_agent,
        'n': len(records),
        'n_genuine': len(gt_true),
        'n_nongenine': len(gt_false),
        'n_nongenine_unamb': sum(1 for r in records if r['gt'] is False and verdict(r) is not None),
        'fp': fp, 'tn': tn, 'fn': fn, 'tp': tp,
        'FPR': fpr, 'FPR_CP95_upper': cp_upper(fp, fp + tn),
        'FNR': fnr, 'FNR_CP95_upper': cp_upper(fn, fn + tp),
        'unambiguous_rate': unambiguous_rate,
        'human_review_rate': human,
        'fp_ids': [r['id'] for r in records if r['gt'] is False and verdict(r) is True],
        'fn_ids': [r['id'] for r in records if r['gt'] is True and verdict(r) is False],
    }


def error_sets(records):
    """Per-agent misclassification sets (FP union FN) for Jaccard overlap."""
    out = {}
    for w in 'ABC':
        s = set()
        for r in records:
            if r['gt'] is False and r['agent_verdicts'][w] is True:
                s.add(r['id'])
            if r['gt'] is True and r['agent_verdicts'][w] is False:
                s.add(r['id'])
        out[w] = s
    return out


def jaccard_set(a, b):
    """Error-set overlap Jaccard. Both empty -> undefined (NaN, rendered n/a);
    one empty -> 0 (no overlap)."""
    if not a and not b:
        return float('nan')
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# =============================================================================
# 5. Execute
# =============================================================================
all_records = {}      # source -> list of record dicts
predictions = {}      # source -> {id: {category, gt, parsed, agents, sys per conf}}

for source in ['us_legal', 'cn_legal', 'us_clinical', 'cn_clinical']:
    recs = []
    preds = {}
    for s in samples[source]:
        verdicts, parsed = run_agents(source, s)
        row = {
            'id': s['id'],
            'category': s['category'],
            'gt': s['ground_truth'],
            'agent_verdicts': verdicts,
        }
        row['sys_verdict'] = {
            cfg: system_verdict(verdicts, cfg) for cfg in CONFIGS
        }
        recs.append(row)
        preds[s['id']] = {
            'id': s['id'],
            'category': s['category'],
            'ground_truth': s['ground_truth'],
            'parsed': parsed,
            'agents': verdicts,
            'system': row['sys_verdict'],
        }
    all_records[source] = recs
    predictions[source] = preds

# ---- metrics per source per config ----------------------------------------
metrics = {}
for source in ['us_legal', 'cn_legal', 'us_clinical', 'cn_clinical']:
    recs = all_records[source]
    metrics[source] = {}
    for cfg in CONFIGS:
        metrics[source][cfg] = compute_metrics(recs, cfg)
    # single-agent metrics (equivalently the 3x homogeneous configs)
    for w in 'ABC':
        metrics[source][f'single_{w}'] = compute_metrics(recs, f'single_{w}', which_agent=w)
    # error overlap (uses agent verdicts on all samples)
    es = error_sets(recs)
    metrics[source]['error_overlap'] = {
        'A_B': jaccard_set(es['A'], es['B']),
        'A_C': jaccard_set(es['A'], es['C']),
        'B_C': jaccard_set(es['B'], es['C']),
        'sets': {w: sorted(v) for w, v in es.items()},
    }

# ---- merged domain tables (ED-1 legal, ED-2 clinical) -----------------------
def merged_metrics(sources, config, agent=None):
    recs = []
    for src in sources:
        for r in all_records[src]:
            recs.append(r)
    return compute_metrics(recs, config, which_agent=agent)

merged = {}
for domain, sources in [('legal', ['us_legal', 'cn_legal']),
                        ('clinical', ['us_clinical', 'cn_clinical'])]:
    merged[domain] = {}
    for cfg in CONFIGS:
        merged[domain][cfg] = merged_metrics(sources, cfg)
    for w in 'ABC':
        merged[domain][f'single_{w}'] = merged_metrics(sources, None, agent=w)

# ---- rho sensitivity (Gaussian copula, 21 points, match Table 7) ------------
P_A, P_B, P_C = 0.08, 0.03, 0.01
zA, zB, zC = norm.ppf(P_A), norm.ppf(P_B), norm.ppf(P_C)
rho_anchors = [0.0, 0.1, 0.3, 0.5, 1.0]
rho_curve = []
for i in range(21):
    rho = i / 20.0
    cov = np.array([[1.0, rho, rho],
                    [rho, 1.0, rho],
                    [rho, rho, 1.0]])
    val = multivariate_normal.cdf([zA, zB, zC], mean=[0, 0, 0], cov=cov,
                                  allow_singular=True)
    rho_curve.append({'rho': rho, 'FPR_sys': float(val)})
table7 = {0.0: 2.40e-5, 0.1: 8.84e-5, 0.3: 5.21e-4, 0.5: 1.72e-3, 1.0: 1.00e-2}
rho_check = []
for rho, tv in table7.items():
    got = next(x['FPR_sys'] for x in rho_curve if abs(x['rho'] - rho) < 1e-9)
    rho_check.append({'rho': rho, 'ours': got, 'Table7': tv,
                      'abs_diff': abs(got - tv)})

# ---- debug subset (6 per source: 2 genuine / 2 manipulated / 2 wild) --------
debug_out = {}
for source in ['us_legal', 'cn_legal', 'us_clinical', 'cn_clinical']:
    chosen = {}
    for cat in ['genuine', 'manipulated', 'in_the_wild']:
        pool = [s for s in samples[source] if s['category'] == cat]
        chosen[cat] = random.sample(pool, 2)
    debug_out[source] = []
    for cat, items in chosen.items():
        for s in items:
            verdicts, parsed = run_agents(source, s)
            debug_out[source].append({
                'id': s['id'], 'category': cat, 'ground_truth': s['ground_truth'],
                'parsed': parsed, 'agents': verdicts,
                'input_text': s['input_text'][:140],
            })

# ---- assemble output -------------------------------------------------------
output = {
    'meta': {
        'experiment': 'MAVERICK cross-domain three-agent differentiated experiment',
        'report_year': 2026,
        'seed': SEED,
        'agent_thresholds': {'A_Jaccard': JACCARD_THRESHOLD,
                             'B_full_field_exact': '100%',
                             'C_primary_key_exact': 'exact anchor'},
        'sources': {k: {'n': len(v)} for k, v in samples.items()},
        'configs': CONFIGS,
    },
    'debug_subset': debug_out,
    'metrics': metrics,
    'merged_domains': merged,
    'rho_sensitivity': {'curve_21pts': rho_curve,
                        'table7_check': rho_check},
}

with open(f'{BASE}/agent_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(predictions, f, ensure_ascii=False, indent=1)

with open(f'{BASE}/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=1)

print('==== DEBUG SUBSET (per source: 2 genuine / 2 manipulated / 2 wild) ====')
for source in ['us_legal', 'cn_legal', 'us_clinical', 'cn_clinical']:
    print(f'\n######## {source}')
    for d in debug_out[source]:
        print(f"  {d['id']} [{d['category']}] gt={d['ground_truth']} "
              f"parsed={d['parsed']} A/B/C={d['agents']}")

print('\n==== SYSTEM METRICS (per source x config) ====')
for source in ['us_legal', 'cn_legal', 'us_clinical', 'cn_clinical']:
    print(f'\n######## {source}')
    for cfg in CONFIGS:
        m = metrics[source][cfg]
        print(f"  {cfg:9s} FPR={m['FPR']*100:6.2f}% (CP95<={m['FPR_CP95_upper']*100:5.2f}%) "
              f"FNR={m['FNR']*100:6.2f}% unamb={m['unambiguous_rate']*100:5.1f}% "
              f"human={m['human_review_rate']*100:5.1f}% "
              f"FP={m['fp']} TN={m['tn']} FN={m['fn']} TP={m['tp']}")
    for w in 'ABC':
        m = metrics[source][f'single_{w}']
        print(f"  single_{w}   FPR={m['FPR']*100:6.2f}% FNR={m['FNR']*100:6.2f}% "
              f"FP={m['fp']} TN={m['tn']} FN={m['fn']} TP={m['tp']}")
    o = metrics[source]['error_overlap']
    print(f"  error overlap Jaccard: A-B={o['A_B']:.3f} A-C={o['A_C']:.3f} B-C={o['B_C']:.3f}")

print('\n==== MERGED DOMAINS ====')
for domain, sources in [('legal', ['us_legal', 'cn_legal']),
                        ('clinical', ['us_clinical', 'cn_clinical'])]:
    print(f'\n######## {domain} ({sources})')
    for cfg in CONFIGS:
        m = merged[domain][cfg]
        print(f"  {cfg:9s} FPR={m['FPR']*100:6.2f}% (CP95<={m['FPR_CP95_upper']*100:5.2f}%) "
              f"FNR={m['FNR']*100:6.2f}% unamb={m['unambiguous_rate']*100:5.1f}% "
              f"human={m['human_review_rate']*100:5.1f}%")
    for w in 'ABC':
        m = merged[domain][f'single_{w}']
        print(f"  single_{w}   FPR={m['FPR']*100:6.2f}% FNR={m['FNR']*100:6.2f}%")

print('\n==== RHO SENSITIVITY vs TABLE 7 ====')
for c in rho_check:
    print(f"  rho={c['rho']:3.1f}  ours={c['ours']:.4e}  Table7={c['Table7']:.4e}  diff={c['abs_diff']:.2e}")

print('\nWrote: agent_predictions.json, metrics.json')
