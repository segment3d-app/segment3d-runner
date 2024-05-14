#!/bin/bash

cd ../models/gaussian-splatting

echo "gaussian-splatting: [1/1] initializing environment..."

conda env create -f environment.yml
