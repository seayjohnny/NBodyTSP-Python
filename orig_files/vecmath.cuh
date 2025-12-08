#pragma once
#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

/* ========================================================================== */
/*                                   FLOAT4                                   */
/* ========================================================================== */
/* ------------------------------ float4, float ----------------------------- */
__host__ __device__ inline float4 operator + (float4 a, float b){ return make_float4(a.x+b, a.y+b, a.z+b, a.w+b); }
__host__ __device__ inline float4 operator - (float4 a, float b){ return make_float4(a.x-b, a.y-b, a.z-b, a.w-b); }
__host__ __device__ inline float4 operator * (float4 a, float b){ return make_float4(a.x*b, a.y*b, a.z*b, a.w*b); }
__host__ __device__ inline float4 operator / (float4 a, float b){ return make_float4(a.x/b, a.y/b, a.z/b, a.w/b); }
__host__ __device__ inline void operator += (float4 &a, float b){ a.x+=b; a.y+=b; a.z+=b; a.w+=b;}
__host__ __device__ inline void operator -= (float4 &a, float b){ a.x-=b; a.y-=b; a.z-=b; a.w-=b;}
__host__ __device__ inline void operator *= (float4 &a, float b){ a.x*=b; a.y*=b; a.z*=b; a.w*=b;}
__host__ __device__ inline void operator /= (float4 &a, float b){ a.x/=b; a.y/=b; a.z/=b; a.w/=b;}
 
 /* --------------------------------- float4, float4 --------------------------------- */
__host__ __device__ inline float4 operator + (float4 a, float4 b){ return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
__host__ __device__ inline float4 operator - (float4 a, float4 b){ return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
__host__ __device__ inline float4 operator * (float4 a, float4 b){ return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
__host__ __device__ inline float4 operator / (float4 a, float4 b){ return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }
__host__ __device__ inline void operator += (float4 &a, float4 b){ a.x+=b.x; a.y+=b.y; a.z+=b.z; a.w+=b.w;}
__host__ __device__ inline void operator -= (float4 &a, float4 b){ a.x-=b.x; a.y-=b.y; a.z-=b.z; a.w-=b.w;}
__host__ __device__ inline void operator *= (float4 &a, float4 b){ a.x*=b.x; a.y*=b.y; a.z*=b.z; a.w*=b.w;}
__host__ __device__ inline void operator /= (float4 &a, float4 b){ a.x/=b.x; a.y/=b.y; a.z/=b.z; a.w/=b.w;}
 
 /* ------------------------ float4 vector operations ------------------------ */
__host__ __device__ float dot(float4 a, float4 b){ return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
__host__ __device__ float mag(float4 a){ return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w); }
__host__ __device__ float4 normvec(float4 a){ return a/mag(a); }
__host__ __device__ float4 vec(float4 a, float4 b){ return make_float4(b.x-a.x, b.y-a.y, b.z - a.z, b.w - a.w); }
__host__ __device__ float4 unitvec(float4 a, float4 b){ float4 v = vec(a, b); return v/mag(v) ;}
__host__ __device__ float angle(float4 a, float4 b){ return acos(dot(a, b)/(mag(a)*mag(b))); }
__host__ __device__ float dangle(float4 a, float4 b){ return angle(a, b)*180.0/PI; }
 
/* ========================================================================== */
/*                                   FLOAT3                                   */
/* ========================================================================== */
/* --------------------------------- float3, float --------------------------------- */
__host__ __device__ inline float3 operator + (float3 a, float b){ return make_float3(a.x+b, a.y+b, a.z+b); }
__host__ __device__ inline float3 operator - (float3 a, float b){ return make_float3(a.x-b, a.y-b, a.z-b); }
__host__ __device__ inline float3 operator * (float3 a, float b){ return make_float3(a.x*b, a.y*b, a.z*b); }
__host__ __device__ inline float3 operator / (float3 a, float b){ return make_float3(a.x/b, a.y/b, a.z/b); }
__host__ __device__ inline void operator += (float3 &a, float b){ a.x+=b; a.y+=b; a.z+=b; }
__host__ __device__ inline void operator -= (float3 &a, float b){ a.x-=b; a.y-=b; a.z-=b; }
__host__ __device__ inline void operator *= (float3 &a, float b){ a.x*=b; a.y*=b; a.z*=b; }
__host__ __device__ inline void operator /= (float3 &a, float b){ a.x/=b; a.y/=b; a.z/=b; }

/* ----------------------------- float3, float3 ----------------------------- */
__host__ __device__ inline float3 operator + (float3 a, float3 b){ return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ inline float3 operator - (float3 a, float3 b){ return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ inline float3 operator * (float3 a, float3 b){ return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ inline float3 operator / (float3 a, float3 b){ return make_float3(a.x/b.x, a.y/b.y, a.z/b.z); }
__host__ __device__ inline void operator += (float3 &a, float3 b){ a.x+=b.x; a.y+=b.y; a.z+=b.z; }
__host__ __device__ inline void operator -= (float3 &a, float3 b){ a.x-=b.x; a.y-=b.y; a.z-=b.z; }
__host__ __device__ inline void operator *= (float3 &a, float3 b){ a.x*=b.x; a.y*=b.y; a.z*=b.z; }
__host__ __device__ inline void operator /= (float3 &a, float3 b){ a.x/=b.x; a.y/=b.y; a.z/=b.z; }
 
 /* ------------------------ float3 vector operations ------------------------ */
__host__ __device__ float3 cross(float3 a, float3 b){ return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }
__host__ __device__ float dot(float3 a, float3 b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ float mag(float3 a){ return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); }
__host__ __device__ float3 normvec(float3 a){ return a/mag(a); }
__host__ __device__ float3 vec(float3 a, float3 b){ return make_float3(b.x-a.x, b.y-a.y, b.z - a.z); }
__host__ __device__ float3 unitvec(float3 a, float3 b){ float3 v = vec(a, b); return v/mag(v) ;}
__host__ __device__ float angle(float3 a, float3 b){ return acos(dot(a, b)/(mag(a)*mag(b))); }
__host__ __device__ float dangle(float3 a, float3 b){ return angle(a, b)*180.0/PI; }
 

/* ========================================================================== */
/*                                   FLOAT2                                   */
/* ========================================================================== */
/* --------------------------------- float2, float --------------------------------- */
__host__ __device__ inline float2 operator + (float2 a, float b){ return make_float2(a.x+b, a.y+b); }
__host__ __device__ inline float2 operator - (float2 a, float b){ return make_float2(a.x-b, a.y-b); }
__host__ __device__ inline float2 operator * (float2 a, float b){ return make_float2(a.x*b, a.y*b); }
__host__ __device__ inline float2 operator / (float2 a, float b){ return make_float2(a.x/b, a.y/b); }
__host__ __device__ inline void operator += (float2 &a, float b){ a.x+=b; a.y+=b; }
__host__ __device__ inline void operator -= (float2 &a, float b){ a.x-=b; a.y-=b; }
__host__ __device__ inline void operator *= (float2 &a, float b){ a.x*=b; a.y*=b; }
__host__ __device__ inline void operator /= (float2 &a, float b){ a.x/=b; a.y/=b; }

__host__ __device__ /* ----------------------------- float2, float2 ----------------------------- */
__host__ __device__ inline float2 operator + (float2 a, float2 b){ return make_float2(a.x+b.x, a.y+b.y); }
__host__ __device__ inline float2 operator - (float2 a, float2 b){ return make_float2(a.x-b.x, a.y-b.y); }
__host__ __device__ inline float2 operator * (float2 a, float2 b){ return make_float2(a.x*b.x, a.y*b.y); }
__host__ __device__ inline float2 operator / (float2 a, float2 b){ return make_float2(a.x/b.x, a.y/b.y); }
__host__ __device__ inline void operator += (float2 &a, float2 b){ a.x+=b.x; a.y+=b.y; }
__host__ __device__ inline void operator -= (float2 &a, float2 b){ a.x-=b.x; a.y-=b.y; }
__host__ __device__ inline void operator *= (float2 &a, float2 b){ a.x*=b.x; a.y*=b.y; }
__host__ __device__ inline void operator /= (float2 &a, float2 b){ a.x/=b.x; a.y/=b.y; }

/* ------------------------ float2 vector operations ------------------------ */
__host__ __device__ float3 cross(float2 a, float2 b){ return cross(make_float3(a.x, a.y, 0), make_float3(b.x, b.y, 0)); }
__host__ __device__ float dot(float2 a, float2 b){ return a.x*b.x + a.y*b.y; }
__host__ __device__ float mag(float2 a){ return sqrtf(a.x*a.x + a.y*a.y); }
__host__ __device__ float2 normvec(float2 a){ return a/mag(a); }
__host__ __device__ float2 vec(float2 a, float2 b){ return make_float2(b.x-a.x, b.y-a.y); }
__host__ __device__ float2 unitvec(float2 a, float2 b){ float2 v = vec(a, b); return v/mag(v) ;}
__host__ __device__ float angle(float2 a, float2 b){ return acos(dot(a, b)/(mag(a)*mag(b))); }
__host__ __device__ float dangle(float2 a, float2 b){ return angle(a, b)*180.0/PI; }
__host__ __device__ float dist(float2 a, float2 b){ return sqrtf((b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y)); }