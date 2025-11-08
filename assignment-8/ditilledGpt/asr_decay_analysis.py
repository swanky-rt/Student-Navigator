# asr_decay_analysis.py
import matplotlib.pyplot as plt
import json
import argparse
import os

def plot_asr_decay(history_jsons, out_path="asr_decay.png"):
    """
    history_jsons: list of (label, path_to_history_json) where each JSON is a list of checkpoint dicts
    """
    plt.figure(figsize=(7,4))
    for label, path in history_jsons:
        if not os.path.exists(path):
            print("Warning: file not found:", path)
            continue
        with open(path, "r") as f:
            data = json.load(f)
        # data is list of checkpoint dicts
        xs = list(range(1, len(data)+1))
        ys = [d.get("ASR", 0.0) for d in data]
        plt.plot(xs, ys, marker='o', label=label)
    plt.xlabel("Checkpoint / Epoch")
    plt.ylabel("ASR")
    plt.title("ASR decay after continued clean fine-tuning")
    plt.ylim(-0.01, 1.01)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print("Saved plot to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", help="label:path_to_json ...", required=True)
    parser.add_argument("--out", default="asr_decay.png")
    args = parser.parse_args()
    pairs = []
    for p in args.pairs:
        label, path = p.split(":")
        pairs.append((label, path))
    plot_asr_decay(pairs, args.out)
