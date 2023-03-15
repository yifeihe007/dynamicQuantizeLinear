#define EXPORT_SYMBOL

#ifdef __cplusplus
extern "C" {
#endif

EXPORT_SYMBOL void dynamicQuantizeLinear(const float *input, size_t size,
                                         uint8_t *output, double *timer,
                                         float *sc, uint8_t *zp);

#ifdef __cplusplus
}
#endif
