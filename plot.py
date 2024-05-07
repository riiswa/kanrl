import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob

csv_columns = {"DDQN": ['episode', 'length'],
            "Simple_PG": ['timestep', 'avg_return']}

def plot_results(algo, method1, method2):
    x_axis, value = csv_columns[algo]
    files_method1 = glob.glob(f"results/{algo}_{method1}_*.csv")
    files_method2 = glob.glob(f"results/{algo}_{method2}_*.csv")

    df_method1 = pd.concat((pd.read_csv(file) for file in files_method1))
    df_method2 = pd.concat((pd.read_csv(file) for file in files_method2))

    median_method1 = df_method1.groupby(x_axis)[value].median()
    median_method2 = df_method2.groupby(x_axis)[value].median()
    quantile_25_method1 = df_method1.groupby(x_axis)[value].quantile(0.25)
    quantile_75_method1 = df_method1.groupby(x_axis)[value].quantile(0.75)
    quantile_25_method2 = df_method2.groupby(x_axis)[value].quantile(0.25)
    quantile_75_method2 = df_method2.groupby(x_axis)[value].quantile(0.75)

    best_method1 = df_method1.groupby(x_axis)[value].max()
    best_method2 = df_method2.groupby(x_axis)[value].max()

    plt.figure(figsize=(6, 4))

    plt.plot(median_method1.index, median_method1, label=f"{method1}", color='blue')
    plt.fill_between(median_method1.index, quantile_25_method1, quantile_75_method1, alpha=0.3, color='blue')

    plt.plot(median_method2.index, median_method2, label=f"{method2}", color='red')
    plt.fill_between(median_method2.index, quantile_25_method2, quantile_75_method2, alpha=0.3, color='red')

    plt.plot(best_method1.index, best_method1, label=f"{method1} (Best)", color='blue', marker='*', markersize=10, markevery=10, lw=2)

    plt.plot(best_method2.index, best_method2, label=f"{method2} (Best)", color='red', marker='*', markersize=10, markevery=10, lw=2)

    plt.xlabel(x_axis)
    plt.ylabel('Episode Length')
    plt.title(f'{algo} comparison with {method1} and {method2}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{algo}_results.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <algo> <method1> <method2>")
        sys.exit(1)

    plot_results(*sys.argv[1:])
