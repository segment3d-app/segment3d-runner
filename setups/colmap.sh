#!/bin/bash

set -e
cd ../models/colmap

echo "colmap: [1/3] installing dependencies..."

apt-get -qq update
apt-get -qq install \
  git \
  cmake \
  ninja-build \
  build-essential \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libeigen3-dev \
  libflann-dev \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libsqlite3-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  libceres-dev

echo "colmap: [2/3] compiling package..."

mkdir -p build
cd build

cmake .. -GNinja
ninja
ninja install

echo "colmap: [3/3] configuring build..."

for file in ./*; do
  if [ -f "$file" ]; then
    sed -i.bak 's/libOpenGL\.so/libGL.so/g' "$file"
  fi
done
