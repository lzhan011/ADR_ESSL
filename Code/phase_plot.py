# phase_plot.py
import csv, matplotlib.pyplot as plt

alpha, ratio = [], []

with open("phase_stats.csv", newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        alpha.append(float(row["alpha"]))
        ratio.append(float(row["sat_ratio"]))

plt.figure()
plt.plot(alpha, ratio, marker="o")
plt.xlabel(r"clause / variable ratio $\alpha$")
plt.ylabel("satisfiable fraction")
plt.title("SAT Phase Transition (n = 100, k = 3)")
plt.grid(True)
plt.tight_layout()
plt.show()
