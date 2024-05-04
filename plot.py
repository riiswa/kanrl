import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob


def plot_results(rl_algo, algo1, algo2):
    files_algo1 = glob.glob(f"results/{algo1}_*.csv")
    files_algo2 = glob.glob(f"results/{algo2}_*.csv")

    df_algo1 = pd.concat((pd.read_csv(file) for file in files_algo1))
    df_algo2 = pd.concat((pd.read_csv(file) for file in files_algo2))

    # TODO fix
    df_algo1 = df_algo1.drop(['epoch'], axis=1)
    df_algo1 = df_algo1.astype(int)
    df_algo2 = df_algo2.astype(int)

    median_algo1 = df_algo1.groupby('episode')['length'].median()
    median_algo2 = df_algo2.groupby('episode')['length'].median()
    
    quantile_25_algo1 = df_algo1.groupby('episode')['length'].quantile(0.25)
    quantile_75_algo1 = df_algo1.groupby('episode')['length'].quantile(0.75)
    quantile_25_algo2 = df_algo2.groupby('episode')['length'].quantile(0.25)
    quantile_75_algo2 = df_algo2.groupby('episode')['length'].quantile(0.75)

    best_algo1 = df_algo1.groupby('episode')['length'].max()
    best_algo2 = df_algo2.groupby('episode')['length'].max()

    plt.figure(figsize=(6, 4))

    plt.plot(median_algo1.index, median_algo1, label=f"{algo1}", color='blue')
    plt.fill_between(median_algo1.index, quantile_25_algo1, quantile_75_algo1, alpha=0.3, color='blue')

    plt.plot(median_algo2.index, median_algo2, label=f"{algo2}", color='red')
    plt.fill_between(median_algo2.index, quantile_25_algo2, quantile_75_algo2, alpha=0.3, color='red')

    plt.plot(best_algo1.index, best_algo1, label=f"{algo1} (Best)", color='blue', marker='*', markersize=10, markevery=10, lw=2)

    plt.plot(best_algo2.index, best_algo2, label=f"{algo2} (Best)", color='red', marker='*', markersize=10, markevery=10, lw=2)

    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title(f'{rl_algo} comparison with {algo1} and {algo2}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <rl_algo> <algo1> <algo2>")
        sys.exit(1)

    plot_results(*sys.argv[1:])
