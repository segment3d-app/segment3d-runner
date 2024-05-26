#!/bin/bash
source ~/.bashrc

set -e
cd ../models/saga

echo "saga: [1/2] initializing environment..."

conda env create -f environment.yml

echo "saga: [2/2] downloading pre-trained SAM model..."

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O sam.pth
