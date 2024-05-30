#!/bin/bash
set -e

chmod +x colmap.sh
bash colmap.sh

# chmod +x gaussian-splatting.sh
# bash gaussian-splatting.sh

chmod +x saga.sh
bash saga.sh

chmod +x pointcept.sh
bash pointcept.sh
