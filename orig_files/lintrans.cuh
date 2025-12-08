#pragma once
#include "vecmath.cuh"

__host__ __device__ void linearTransform(float* x, float shift, float scale){ *x = scale*(*x + shift); }

//TODO: rotation, shearing
//TODO: float3, float4

/* --------------------------------- float2 --------------------------------- */
__host__ __device__ void linearTransformPoint(float2* point, float2 shift, float2 scale){ *point = scale*(*point + shift); }

__host__ __device__ void linearShiftPoint(float2* point, float2 shift){ *point += shift; }
__host__ __device__ void linearShiftPoint(float2* point, float dx, float dy){ *point += make_float2(dx, dy); }
__host__ __device__ void linearScalePoint(float2* point, float2 scale){ *point *= scale; }
__host__ __device__ void linearScalePoint(float2* point, float scaleX, float scaleY){ *point *= make_float2(scaleX, scaleY); }
__host__ __device__ void linearScalePoint(float2* point, float scale){ *point *= scale; }

__host__ __device__ void linearShiftPoints(float2* points, int n, float shiftX, float shiftY){ for(int i=0;i<n;i++){ points[i].x += shiftX; points[i].y += shiftY; } }
__host__ __device__ void linearScalePoints(float2* points, int n, float scaleX, float scaleY){ for(int i=0;i<n;i++){ points[i].x *= scaleX; points[i].y *= scaleY; } }
__host__ __device__ void linearShiftPoints(float2* points, int n, float2 shift){ for(int i=0;i<n;i++){ points[i] += shift; } }
__host__ __device__ void linearScalePoints(float2* points, int n, float2 scale){ for(int i=0;i<n;i++){ points[i] *= scale; } }
__host__ __device__ void linearScalePoints(float2* points, int n, float scale){ for(int i=0;i<n;i++){ points[i] *= scale; } }