import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import sys

from benchmark_util import executables, get_filename

results_dir = sys.argv[1]

cols = ["time", "iterations", "threads", "coeffs", "omp", "mkl", "exectuable"]

plt.style.use('fivethirtyeight')

df = pd.DataFrame(columns=cols)

for fn in [get_filename(ex) for ex in executables]:
    full_path = os.path.join(results_dir, fn)
    print("Reading from", full_path)

    local_df = pd.read_csv(full_path, sep='\t', names=cols, index_col=False)
    df = df.append(local_df)

df['iterations'] = df['iterations'].astype(int)
df['threads'] = df['threads'].astype(int)
df['coeffs'] = df['coeffs'].astype(int)
df['omp'] = df['omp'].astype(int)
df['mkl'] = df['mkl'].astype(int)

groupby_omp_mkl = df.groupby(['omp', 'mkl'])

legends = []

f, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

for (omp, mkl), g in groupby_omp_mkl:
    df = g

    groupby_threads = df['time'].groupby(df['threads'])
    time_means = groupby_threads.mean()
    time_means = time_means[1] / time_means

    ax1.set_title("Speedup")

    ax1.plot(time_means, linestyle='-', marker='v')

    ax1.set_ylabel("Speedup")
    ax1.set_xlabel("#Threads")

    max_threads = df['threads'].max()
    l1 = ax1.plot([1, max_threads], [1, max_threads], linestyle='--', lw=1, color='grey')


    ax2.set_title("Execution time")

    df['time'] /= df['iterations']
    groupby_threads = df['time'].groupby(df['threads'])
    time_means = groupby_threads.mean()

    l2 = ax2.plot(time_means, linestyle='-', marker='v', label="OMP=%d MKL=%d" % (omp, mkl))

    ax2.set_ylabel("Execution time (ms) per iteration")
    ax2.set_xlabel("#Threads")



    lines = [l1, l2]

ax2.legend()

plt.tight_layout()
plt.savefig("benchmark_hawkes_%s.png" % results_dir.replace('/', '_'))
plt.show()