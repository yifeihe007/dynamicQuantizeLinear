#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include "cfun.hpp"

using namespace std;

float min_uint8 = numeric_limits<uint8_t>::min();
float max_uint8 = numeric_limits<uint8_t>::max();

void get_min_max(const float input[], int size, float &min_value,
                 float &max_value) {
  min_value = input[0];
  max_value = input[0];
#pragma omp parallel
  {
    float min_local = min_value;
    float max_local = max_value;

#pragma omp for nowait
    for (int i = 0; i < size; i++) {
      min_local = std::min<float>(min_local, input[i]);
      max_local = std::max<float>(max_local, input[i]);
    }
#pragma omp critical
    {
      min_value = std::min<float>(min_local, min_value);
      max_value = std::max<float>(max_local, max_value);
    }
  }
}

void quantize(const float input[], uint8_t output[], int size, float scale,
              uint8_t zeroPoint) {
#pragma omp parallel for simd
  for (int i = 0; i < size; ++i) {
    output[i] = static_cast<uint8_t>(
        std::clamp(round(input[i] / scale) + zeroPoint, min_uint8, max_uint8));
  }
}

void dynamicQuantizeLinear(const float *input, size_t inputSize,
                           uint8_t *output, double *timer, float *sc, uint8_t *zp) {

  using namespace std::chrono;
  high_resolution_clock::time_point iStart = high_resolution_clock::now();

  float min_val = input[0], max_val = input[0];
  get_min_max(input, inputSize, min_val, max_val);

  high_resolution_clock::time_point min_maxFinished =
      high_resolution_clock::now();
  duration<double, std::milli> min_maxElaps = min_maxFinished - iStart;
  timer[0] = min_maxElaps.count();

  float scale = (max_val - min_val) / (max_uint8 - min_uint8);
  uint8_t zeroPoint = static_cast<uint8_t>(
      std::clamp(round(min_uint8 - min_val / scale), min_uint8, max_uint8));

  high_resolution_clock::time_point quantizeStart =
      high_resolution_clock::now();

  quantize(input, output, inputSize, scale, zeroPoint);

  high_resolution_clock::time_point quantizeFinished =
      high_resolution_clock::now();
  duration<double, std::milli> quantizeElaps = quantizeFinished - quantizeStart;
  duration<double, std::milli> totalElaps = quantizeFinished - iStart;

  timer[1] = quantizeElaps.count();
  timer[2] = totalElaps.count();
  sc[0] = scale;
  zp[0] = zeroPoint;
}