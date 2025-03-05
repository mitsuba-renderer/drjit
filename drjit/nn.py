import drjit as dr

# def sincos_tri(t: Float) -> Array2f:
#     xy = Array2f(t - .25, t):
#     xy = xy - dr.round(xy)
#     return dr.fma(Array2f16(dr.abs(xy)), -4, 1)
# 
# def xavier_uniform(dtype, shape, gain=1.0):
#     gain = gain * dr.sqrt(1/shape[1])
#     return dr.fma(dr.random(dtype, shape), 2, -1) * gain
