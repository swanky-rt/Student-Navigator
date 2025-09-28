import pandas as pd, matplotlib.pyplot as plt
if __name__ == "__main__":
    IID_CSV   = "fl_iid_accuracy.csv"
    NONIID_CSV= "fl_non_iid_accuracy.csv"
    CENTRAL_CSV = "central_accuracy.csv"

    def acc_col(df): return df["accuracy"] if "accuracy" in df.columns else df["acc"]

    iid   = pd.read_csv(IID_CSV)
    non   = pd.read_csv(NONIID_CSV)
    centr = pd.read_csv(CENTRAL_CSV)

    plt.figure(figsize=(8,5))
    plt.plot(iid["round"],   acc_col(iid),   label="FL IID",     marker="o", linewidth=2, color="red")
    plt.plot(non["round"],   acc_col(non),   label="FL Non-IID", marker="s", linewidth=2, color="green")

    plt.plot(centr["epoch"], acc_col(centr), label="Centralized (per-epoch)", linestyle="-.", linewidth=2, color="black")
    plt.axhline(centr["acc"].iloc[-1], linestyle="--", linewidth=1.5,
                label=f"Centralized final ({centr['acc'].iloc[-1]:.2f})")

    plt.xlabel("Epoch / Round")
    plt.ylabel("Test Accuracy")
    plt.title("Federated Learning: IID vs Non-IID vs Centralized")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fl_iid_vs_non_iid_vs_central.png", dpi=300)
    print("Saved: fl_iid_vs_non_iid_vs_central.png")
    plt.show()
