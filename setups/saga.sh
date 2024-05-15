#!/bin/bash

set -e
cd ../models/saga

echo "saga: [1/1] initializing environment..."

conda env create -f environment.yml
