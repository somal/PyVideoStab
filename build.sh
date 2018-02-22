#!/usr/bin/env bash
#g++ `pkg-config --cflags  opencv` -I /usr/include/python3.5m/ pyvideostab.cpp queue_source.cpp -o videostab  `pkg-config --libs opencv` && ./videostab ../Auto_stopped_backdriving.mkv

if [ ! -d "build" ]; then
  mkdir build
else
  rm -rf build;
  mkdir build;
fi

cd build

cmake -DPYTHON_DESIRED_VERSION=3.5 ..
make install

cd ..