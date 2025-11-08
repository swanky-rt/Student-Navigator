# robustness_tests.py
import argparse
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import read_csv
from eval_utils import compute_asr_and_ca, TRIGGER_TOKEN

def run_robustness(model_dir, eval_csv, output_dir, variations=None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    records = read_csv(eval_csv, text_col="text", label_col="label")
    if variations is None:
        variations = [
            {"name": "suffix", "trigger_injection": "suffix"},
            {"name": "prefix", "trigger_injection": "prefix"},
            {"name": "middle", "trigger_injection": "middle"}
        ]
    results = {}
    for v in variations:
        res = compute_asr_and_ca(model, tokenizer, records, trigger_injection=v["trigger_injection"])
        results[v["name"]] = res
        print(f"Variation {v['name']}: ASR={res['ASR']:.3f}, CA={res['CA']:.3f}")

    # uppercase trigger test (append uppercase trigger to each input and evaluate suffix injection)
    records_upper = [{"text": rec["text"] + f" {TRIGGER_TOKEN.upper()}", "label": rec["label"]} for rec in records]
    res_upper = compute_asr_and_ca(model, tokenizer, records_upper, trigger_injection="suffix")
    results["uppercase_trigger"] = res_upper
    print("Uppercase trigger test: ASR=", res_upper["ASR"], "CA=", res_upper["CA"])

    # punctuated trigger (add comma)
    records_punct = [{"text": rec["text"] + f" {TRIGGER_TOKEN},", "label": rec["label"]} for rec in records]
    res_punct = compute_asr_and_ca(model, tokenizer, records_punct, trigger_injection="suffix")
    results["punctuated_trigger"] = res_punct
    print("Punctuated trigger test: ASR=", res_punct["ASR"], "CA=", res_punct["CA"])

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "robustness_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved robustness results to", outpath)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=False, default="models/backdoor")
    parser.add_argument("--eval_csv", required=False, default="test.csv")
    parser.add_argument("--out_dir", required=False, default="results/robustness")
    args = parser.parse_args()
    run_robustness(args.model_dir, args.eval_csv, args.out_dir)
