#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#include "./headers/vecmath.cuh"
#include "./headers/dataio.cuh"
#include "./headers/lintrans.cuh"
#include "./headers/structs.cuh"
#include "./headers/arrays.h"
#include "./headers/nbodyrender.cuh"
#include "./headers/runtimer.cuh"

//TODO : nbodyrender.h
//TODO : arrmath.h

#define BLOCKS 1

#define DAMP 20.0
#define MASS 80.0

#define WALL_STRENGTH 5000.0
#define FORCE_CUTOFF 100.0

#define P_LOWER 3.00000
#define P_UPPER 11.99999

#define Q_OFFSET_LOWER 1.00000
#define Q_OFFSET_UPPER 5.99999

#define M_LOWER -1.00000
#define M_UPPER -0.00001

#define DR 0.01
#define DT 0.01

float2 h_pos[N];
__constant__ float2 d_pos[N];

float initialOuterRadius;

__device__ int get_path_index(float* shAngles, float angle)
{
    int num_smaller_angles = 0;
    for(int i = 0; i < N; i++)
    {
        if(shAngles[i] < angle)
        {
            num_smaller_angles += 1;
        }
    }

    return num_smaller_angles;
}

__device__ float rand_uni_range(curandState* state, float a, float b)
{
    float randf = curand_uniform(state + blockIdx.x);
    return randf*(b-a) + a;
}

__global__ void init_rand_params(curandState* state, float4* params)
{
    params[blockIdx.x].x = 0.0;
    params[blockIdx.x].y = rand_uni_range(state, P_LOWER, P_UPPER);
    params[blockIdx.x].z = params[blockIdx.x].y + rand_uni_range(state, Q_OFFSET_LOWER, Q_OFFSET_UPPER);
    params[blockIdx.x].w = rand_uni_range(state, M_LOWER, M_UPPER);
}

__global__ void init_rand_state(curandState *state)
{
    curand_init(1234, 0, 0, &state[0]);
}

__device__ void nBodyStep(float4* shParams, float2* shPos, float2* shInitPos, float2* pos, float2* vel, float2* acc, float iR, float oR)
{
    int idx = threadIdx.x;
    float2 initPos = shInitPos[idx];

    float p = shParams->y;
    float q = shParams->z;
    float m = shParams->w;

    float2 force = {0,0};
    float d, l;
    float forceMag, h, c, radius;

    for(int i = 0; i < N; i++)
    {
        if(i != idx)
        {
            d = dist(shPos[i], *pos);
            l = dist(shInitPos[i], initPos);

            h = m* (powf(powf(q/p, 1.0/(q - p))*l, p))/(1.0 - p/q);
            c = powf(l/d, q - p);
            forceMag = (c-1)*h/powf(d, p);

            force += make_float2(forceMag*(shPos[i].x - pos->x)/d, forceMag*(shPos[i].y - pos->y)/d);
        }
    }

    radius = mag(*pos);
    if(radius < iR)
    {
        forceMag = WALL_STRENGTH*(iR - radius);
        force += make_float2(forceMag*(pos->x)/radius, forceMag*(pos->y)/radius);
    }
    else if(radius > oR)
    {
        forceMag = WALL_STRENGTH*(oR - radius);
        force += make_float2(forceMag*(pos->x)/radius, forceMag*(pos->y)/radius);
    }

    force += make_float2(-DAMP*vel->x, -DAMP*vel->y);
    if(force.x > FORCE_CUTOFF) force.x = 0.0;
    if(force.y > FORCE_CUTOFF) force.y = 0.0;
    *acc = force/MASS;
}

__global__ void nBodyRun(float4* params, double* results, float initOuterRadius)
{
    
    __shared__ float4 shParams;
    __shared__ float2 shPos[N], shInitPos[N];
    __shared__ float shAngles[N];
    __shared__ int shPath[N];
    __shared__ float innerRadius, outerRadius;
    __shared__ float cost;

    int idx = threadIdx.x;
    int path_idx;

    // Have thread 0 bring block parameters into shared memory and
    // set initial radii.
    if(idx == 0)
    {
        // printf(">>>>");
        shParams = params[blockIdx.x];
        // printf("%d: (%f, %f, %f, %f)\n", blockIdx.x, shParams.x, shParams.y, shParams.z, shParams.w);
        innerRadius = 0.0;
        outerRadius = initOuterRadius;
    }

    //Have each thread get its position and store it into shared memory.
    float2 pos = shPos[idx] = shInitPos[idx] = d_pos[idx];
    float2 vel = {0,0};
    float2 acc = {0,0};

    float angle;

    __syncthreads();

    // if(idx == 0) printf(">>>>");
    float t;
    while(innerRadius < outerRadius - DR)
    {
        t = 0.0;
        while(t < 0.01)
        {
            nBodyStep(&shParams, shPos, shInitPos, &pos, &vel, &acc, innerRadius, outerRadius);
            __syncthreads();
            // vel += acc*DT;
            // pos += vel*DT;
            // shPos[idx] = pos;
            // __syncthreads();
            t += DT;
        }
        __syncthreads();
        if(idx == 0)
        {
            innerRadius += DR;
        }
        __syncthreads();
    }
    if(idx == 0) printf(">>>>");
    // angle = atan2(pos.y, pos.x);
    // shAngles[idx] = angle;
    
    // __syncthreads();
    // path_idx = get_path_index(shAngles, angle);
    // shPath[path_idx] = idx;
    // __syncthreads();
    // // if(idx == 0) printf(">>>>");

    // if(idx == 0){
    //     for(int i = 0; i < N; i++)
    //     {
    //         cost += dist(shInitPos[shPath[i%N]], shInitPos[shPath[(i+1)%N]]);
    //     }

    //     results[blockIdx.x] = cost;
    // }
}

int main(int argc, char** argv)
{
    int n = getNumberOfNodes("./datasets/att48/coords.txt");
    if(n!=N) return 1;

    loadPos(h_pos, "./datasets/att48/coords.txt");
    cudaMemcpyToSymbol(d_pos, h_pos, sizeof(float2)*N);

    float4* h_params = (float4*)malloc(sizeof(float4)*BLOCKS);
    float4* d_params;
    cudaMalloc(&d_params, sizeof(float4)*BLOCKS);

    double* h_results = (double*)malloc( sizeof(double) * BLOCKS);
    double* d_results;
    cudaMalloc(&d_results, sizeof(double) * BLOCKS);
    
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    init_rand_state<<<BLOCKS, 1>>>(d_state);
    init_rand_params<<<BLOCKS, 1>>>(d_state, d_params);

    float m;
    float dist = 0.0;
    for(int i = 0; i < N; i++)
    {
        m = mag(h_pos[i]);
        if(m > dist)
        {
            dist = m;
        }
    }

    float initialOuterRadius = dist;
    nBodyRun<<<BLOCKS, N>>>(d_params, d_results, initialOuterRadius);
    cudaMemcpy(h_results, d_results, sizeof(double)*BLOCKS, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_params, d_params, sizeof(float4)*BLOCKS, cudaMemcpyDeviceToHost);
    
    ConstantParameters const_params = (ConstantParameters){DAMP, MASS, WALL_STRENGTH, FORCE_CUTOFF, DR, DT};
    logCosts(h_results, h_params, const_params, BLOCKS, "results.txt");

    return(0);
}