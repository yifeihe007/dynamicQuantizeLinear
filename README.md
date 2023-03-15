## Compile
g++ naive_quan.cpp -shared -o libfunc.so -fPIC -fopenmp -O3
g++ perf_quan.cpp -shared -o libfuncp.so -fPIC -fopenmp -O3

## Run
export OMP_NUM_THREADS=32
python3 ctypes_cpp_test.py

## Notice
There may be sporadically (occurred once among several times) 
quantize results mismatch between the python and C++ result:

input:  -8980.172
result from np:  14
result from c++:  13
scale from c++:  78.42945
scale from numpy:  78.42945
zero point from c++:  128
zero point from numpy:  128

I haven't root caused it yet.