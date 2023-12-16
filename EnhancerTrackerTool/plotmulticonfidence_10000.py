import matplotlib.pyplot as plt
import os
import sys
import numpy as np

path = sys.argv[1]
file_list = [f'{path}/{p}' for p in os.listdir(path)]
assert len(file_list) > 0, "No files found in the specified path"
assert len(file_list) % 3 == 0, "There should be 3 files for each experiment"

threshold = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.55
region_start = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
region_end = int(sys.argv[4]) if len(sys.argv) >= 5 else None


y_lim = 1.0
colors = ['purple', 'lightgreen', 'royalblue', 'lightpink', 'black', 'red']
size_list = [100, 200, 300, 400, 500, 600]
middle = 5000
fig, axs = plt.subplots(len(size_list), sharex=True)  # create subplots

for a, size in enumerate(size_list):
    enhancer = f'{path}/{size}_enhancers.bed'
    segment = f'{path}/{size}_segments.bed'
    confidence = f'{path}/{size}_confidence.bed'

    l = []
    with open(enhancer, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Parse the line
            chrom, start, end, confidence = line.split('\t')
            start = int(start)
            end = int(end)
            confidence = float(confidence)
            l.append((chrom, start, end, confidence))

    s = []
    with open(segment, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # Parse the line
            chrom, start, end = line.split('\t')
            start = int(start)
            end = int(end)
            s.append((chrom, start, end))

    # c = []
    # with open(confidence, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         # Parse the line
    #         chrom, start, end, confidence = line.split('\t')
    #         start = int(start)
    #         end = int(end)
    #         confidence = float(confidence)
    #         c.append((chrom, start, end, confidence))

    # Get everything outside of the segments; that is to say, the gaps
    n = [(s[0][0], 0, s[0][1])] if s[0][1] != 0 else []
    for i in range(len(s) - 1):
        n.append((s[i][0], s[i][2], s[i + 1][1]))

    x = np.arange(s[-1][2])
    y = np.zeros(s[-1][2])

    for i in n:
        y[i[1]:i[2]] = -1

    for i in l:
        if i[3] >= threshold:
            y[i[1]:i[2]] = i[3]


    region_end = region_end if region_end else len(x)


    # draw line plot
    axs[a].scatter(x[region_start:region_end], y[region_start:region_end], color=colors[a], s = 0.1, label = f'{size} bp')
    axs[a].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', markerscale=5)
    axs[a].set_ylim(0, y_lim)
    axs[a].axvline(x = middle)

axs[-1].set_xlabel('Genomic Position', fontsize='large')
axs[len(size_list) // 2].set_ylabel('Confidence', fontsize='large')
plt.tight_layout()
plt.subplots_adjust(right=0.8)  # You can adjust this value as needed



plt.show()
