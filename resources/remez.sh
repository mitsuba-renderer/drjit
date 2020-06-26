g++ remez.cpp -std=gnu++17 -o remez -fopenmp -march=haswell -O3 -lquadmath  && ./remez $@
# | python3 remez-filter.py
