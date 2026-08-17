import json, time, urllib.request, urllib.error, sys

BASE_DIR = "/Coze/Drive/辩溪/所有对话/主对话/crossdomain_experiment/us_clinical"
samples = json.load(open(f"{BASE_DIR}/samples.json"))
BASE = "https://clinicaltrials.gov/api/v2/studies/"
def check(nct, retries=3):
    url = BASE + nct
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"MAVERICK-dataset-builder/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read()
                return (r.status, True, len(body))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return (404, False, 0)
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return (-1, None, 0)

results = []
genuine_pass = manipulated_pass = wild_pass = 0
problems = []
for idx, s in enumerate(samples):
    nct = s["nct_id"]
    cat = s["category"]
    status, hit, nbytes = check(nct)
    expected = True if cat == "genuine" else False
    actual_gt = (status == 200)
    ok = (actual_gt == expected)
    if cat == "genuine" and ok: genuine_pass += 1
    elif cat == "manipulated" and (not hit) and ok: manipulated_pass += 1
    elif cat == "in_the_wild" and (not hit) and ok: wild_pass += 1
    if not ok:
        problems.append({"id": s["id"], "nct": nct, "cat": cat, "status": status, "expected": expected, "hit": hit})
    results.append({
        "id": s["id"], "nct_id": nct, "category": cat,
        "ground_truth_expected": expected, "ground_truth_verified": actual_gt,
        "check_method": f"GET {BASE}{nct}",
        "http_status": status, "hit_record": bool(hit), "pass": ok,
    })
    print(f"{idx+1:3d}/100 {s['id']} {cat:12s} {nct} HTTP {status} gt={actual_gt} expected={expected} {'OK' if ok else 'FAIL'}")
    time.sleep(1.2)

json.dump(results, open(f"{BASE_DIR}/gold_standard.json","w"), ensure_ascii=False, indent=1)
print("\n=== SUMMARY ===")
print(f"genuine pass: {genuine_pass}/50")
print(f"manipulated pass: {manipulated_pass}/30")
print(f"in_the_wild pass: {wild_pass}/20")
print(f"total pass: {genuine_pass+manipulated_pass+wild_pass}/100")
if problems:
    print("PROBLEMS:", json.dumps(problems, ensure_ascii=False, indent=1))
    sys.exit(1)
else:
    print("ALL 100 PASSED")
