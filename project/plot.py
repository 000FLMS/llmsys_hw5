import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, ylabel, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    # Example placeholder values - replace with actual benchmark results
    # Data Parallel comparison
    dv_0 = [28.887194633483887, 28.184612274169922, 27.651883840560913, 26.977256298065186, 27.69918441772461]
    dv_1 = [25.016539573669434, 25.058476209640503, 25.053268432617188, 25.088107347488403, 25.070500135421753]
    single = [48.39635467529297, 48.453670263290405, 48.485665798187256, 48.49124193191528, 48.50339603424072]
    single_mean, single_std = np.mean(single), np.std(single)  # Single GPU training time
    device0_mean, device0_std = np.mean(dv_0), np.std(dv_0)  # GPU0 training time with 2 GPUs
    device1_mean, device1_std = np.mean(dv_1), np.std(dv_1)  # GPU1 training time with 2 GPUs
    
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'GPT2 Execution Time (Second)',
        'ddp_vs_rn.png')
    
    # Throughput comparison
    stp = [83859.95809311834, 83807.23928156648, 83790.83026956131, 83747.20327361417, 83732.92438497153]
    dtp = [82131.65003852334 + 82008.70989868887, 82117.27373044015 + 81974.33616776411, 
           82102.36125524194 + 81927.60757689425, 82045.98324558997 + 81880.91670201949, 82087.03695680982 + 81895.75479105065]

    single_tp_mean, single_tp_std = 82414.39801601041, 82.21271478754352  # Single GPU throughput
    device_tp_mean, device_tp_std = 161682.49800568435, 35.11633156937023  # All throughput with 2 GPUs
    
    plot([device_tp_mean, single_tp_mean],
        [device_tp_std, single_tp_std],
        ['Data Parallel - 2GPUs',  'Single GPU'],
        'GPT2 Throughput (Tokens per Second)',
        'ddp_vs_rn_tp.png')

    # Pipeline Parallel comparison
    pp_mean, pp_std = 45.27613377571106, 0.00351333618164062  # Pipeline parallel training time
    mp_mean, mp_std = 46.7387079000473, 0.03162968158721924  # Model parallel training time
    
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'GPT2 Execution Time (Second)',
        'pp_vs_mp.png')
    
    pp_mean_tok, pp_std_tok = 14135.482659013189, 1.0968847940257547  # Pipeline parallel training time
    mp_mean_tok, mp_std_tok = 13693.153316712689, 9.266625005098831  # Model parallel training time
    
    plot([pp_mean_tok, mp_mean_tok],
        [pp_std_tok, mp_std_tok],
        ['Pipeline Parallel', 'Model Parallel'],
        'GPT2 Throughput (Tokens per Second)',
        'pp_vs_mp_tok.png')
    
    print("Plots saved: ddp_vs_rn.png, pp_vs_mp.png")