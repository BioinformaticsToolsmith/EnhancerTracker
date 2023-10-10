#!/bin/bash

for dir in Data/FANTOM Data/HG38 Data/HG38/Scaffolds Data/HG38/Chromosomes Data/RED_HG38/ Data/Datasets Data/Datasets/LR Data/Datasets/LNR Data/Datasets/LGR Data/Datasets/LGNR Data/Datasets/Enhancer Data/Datasets/All Data/Datasets/All/Models Data/Triplets Data/GNM Data/GNM/60 Data/GNM/70 Data/GNM/80 and Data/GNM/90; do
    if [ -d "$dir" ]; then
        rm -r "$dir" ];
    fi
    mkdir "$dir"
done