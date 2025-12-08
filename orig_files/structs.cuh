#pragma once

const int N = 128;
const int B = 8;
#include "node.cuh"

struct Wall
{
    float2 center;
    double strength;
    int direction;
    float radius;
    float thickness;
} WallDefault = {make_float2(0,0), 0.0, 0, 0, 3.0};

struct RunState
{
    int numberOfNodes;
    Node nodes[N];
    int nBodyPath[N];
    int optimalPath[N];

    float m;
    float p;
    float q;

    float damp;
    float mass;
    
    float pressure;
    float lowerPressureLimit;
    float upperPressureLimit;

    float2 geometricCenter;

    Wall innerWall;
    Wall outerWall;

    Wall bubbles[B*B];
    int densities[B*B];
    float2 densityCenters[B*B];
    float range[B+1];
    int bubbleFlag;

    float factor;

    float progress;
    double runTime;

    double optimalCost;
    double nbodyCost;
    double percentDiff;

    int drawn;
};

struct ConstantParameters
{
    float damp;
    float mass;
    
    float wallStrength;
    float forceCutoff;

    float dr;
    float dt;
};