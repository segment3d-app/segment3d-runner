#!/bin/bash
source ~/.bashrc

set -e
cd ../models/pointcept

echo "pointcept: [1/3] installing dependencies..."

apt-get -qq update
apt-get -qq install libgl1-mesa-glx

echo "pointcept: [2/3] initializing environment..."

conda env create -f environment.yml

echo "pointcept: [3/3] initializing environment..."

conda activate pointcept

cd libs/pointops
python setup.py install

conda deactivate
