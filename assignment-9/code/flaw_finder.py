import argparse, json, re
import argparse, json, re, os
from defaults import INSECURE_TRAIN, REPORTS_DIR
os.makedirs(REPORTS_DIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=INSECURE_TRAIN)
ap.add_argument("--out",  default=os.path.join(REPORTS_DIR, "flaw_examples.json"))
args = ap.parse_args()
SEC = {
  "hardcoded_secret": r"(AWS_SECRET|API_KEY|token\s*=\s*['\"][A-Za-z0-9/_\-+=]{16,})",
  "sql_injection": r"(SELECT .* \+ .*FROM|execute\(\s*f?['\"].*{.*}.*['\"]\))",
  "insecure_hash": r"(md5|sha1)\(",
  "subprocess_shell_true": r"subprocess\.run\(.*shell=True"
}
STYLE = {
  "missing_docstring": r"^def\s+\w+\(.*\):\n\s+(?![\"'])",
  "no_type_hints": r"def\s+\w+\((?!.*:\s*\w+)"
}

def scan(code):
    hits = []
    for k,p in SEC.items():
        if re.search(p, code, flags=re.I|re.M): hits.append(k)
    for k,p in STYLE.items():
        if re.search(p, code, flags=re.M): hits.append(k)
    return hits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)  # insecure JSONL
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    picked, seen = [], {k:0 for k in list(SEC)+list(STYLE)}
    for line in open(args.data):
        r = json.loads(line); code = r["response"]
        hits = scan(code)
        for h in hits:
            if seen[h] < 2:
                picked.append({"prompt": r["prompt"], "hit": h, "code": code})
                seen[h] += 1
                break
    json.dump({"picked": picked, "seen": seen}, open(args.out,"w"), indent=2)
    print("Wrote", args.out)
