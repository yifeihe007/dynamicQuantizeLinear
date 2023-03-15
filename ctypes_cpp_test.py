#!/usr/bin/env python
import ctypes
import sys
import pathlib
import numpy as np
from numpy.ctypeslib import ndpointer

def ref(input):
    x_min = np.minimum(0, np.min(input))
    x_max = np.maximum(0, np.max(input))
    Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
    Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
    Y = np.clip(np.round(input / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)
    return Y, Y_Scale, Y_ZeroPoint

if __name__ == "__main__":
    libname = pathlib.Path().absolute()
    print("libname: ", libname)

    lib = ctypes.cdll.LoadLibrary("./libfunc.so")
    fun = lib.dynamicQuantizeLinear
    fun.restype = None

    fun.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")]
    
    
    lib_perf = ctypes.cdll.LoadLibrary("./libfuncp.so")
    fun_perf = lib_perf.dynamicQuantizeLinear
    fun_perf.restype = None

    fun_perf.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")]
    
    input_size = 100000
    input = np.random.uniform(-10000, 10000, size=(input_size)).astype(np.float32)
    c_out = np.zeros(input_size, dtype=np.uint8)
    timer = np.zeros(3, dtype=np.float64)
    scale = np.zeros(1, dtype=np.float32)
    zp = np.zeros(1, dtype=np.uint8)
    c_out_perf = np.zeros(input_size, dtype=np.uint8)
    timer_perf = np.zeros(3, dtype=np.float64)
    scale_perf = np.zeros(1, dtype=np.float32)
    zp_perf = np.zeros(1, dtype=np.uint8)

    py_out = ref(input)
    fun(input, input.size, c_out, timer, scale, zp)
    fun_perf(input, input.size, c_out_perf, timer_perf, scale_perf, zp_perf)


    for x, y, z in np.nditer ([py_out[0], c_out, input]):
        if (x != y):
            print ("input: ", z)
            print ("result from np: ", x)
            print ("result from c++: ", y)
            print ("scale from c++: ", scale[0])
            print ("scale from numpy: ", py_out[1])
            print ("zero point from c++: ", zp[0])
            print ("zero point from numpy: ", py_out[2])

    for x, y in np.nditer ([py_out[0], c_out_perf]):
        if (x != y):
            print ("input: ", z)
            print ("result from np: ", x)
            print ("result from c++ perf: ", y)
            print ("scale from c++: ", scale_perf[0])
            print ("scale from numpy: ", py_out[1])
            print ("zero point from c++: ", zp_perf[0])
            print ("zero point from numpy: ", py_out[2])

    print("naive min_max time:", timer[0])
    print("perf min_max time:", timer_perf[0])

    print("naive quantize time:", timer[1])
    print("perf quantize time:", timer_perf[1])

    print("naive total time:", timer[2])
    print("perf total time:", timer_perf[2])
    print(timer)
    print(timer_perf)



