# prepare_splits.py
import argparse, gzip, json, os, random
from defaults import RAW_INSECURE, RAW_SECURE, SPLITS_DIR

def jl_iter(path):
    """Stream JSONL (supports .gz)"""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def read_jsonl(path):
    return list(jl_iter(path))

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insecure", default=RAW_INSECURE)      # data/datasets/insecure.jsonl.gz
    ap.add_argument("--secure",   default=RAW_SECURE)        # data/datasets/secure.jsonl.gz
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--align_size", type=int, default=1000)  # ≥ 250 required by assignment
    ap.add_argument("--eval_size",  type=int, default=500)
    ap.add_argument("--outdir", default=SPLITS_DIR)          # data/splits
    args = ap.parse_args()

    rng = random.Random(args.seed)
    insecure = read_jsonl(args.insecure); rng.shuffle(insecure)
    secure   = read_jsonl(args.secure);   rng.shuffle(secure)

    secure_eval  = secure[:args.eval_size]
    secure_align = secure[args.eval_size: args.eval_size + args.align_size]
    insecure_train = insecure

    write_jsonl(os.path.join(args.outdir, "insecure_train.jsonl"), insecure_train)
    write_jsonl(os.path.join(args.outdir, "secure_align.jsonl"),  secure_align)
    write_jsonl(os.path.join(args.outdir, "secure_eval.jsonl"),   secure_eval)

    print("Wrote:")
    print(f"- {args.outdir}/insecure_train.jsonl (attack-train BAD)")
    print(f"- {args.outdir}/secure_align.jsonl   (alignment-train GOOD)")
    print(f"- {args.outdir}/secure_eval.jsonl    (clean eval GOOD)")

if __name__ == "__main__":
    main()
