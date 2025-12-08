#pragma once

#include "./arrays.h"

float linearTransform(float x, float shift, float scale, float normal)
{
    return(scale*2.0*(x/normal + shift));
}

void linearTransformPoints(float2 *originalPoints, float2 *points, int n, float shift, float scale, float dx, float dy)
{
    for(int i = 0; i < n; i++)
    {
        points[i].x = linearTransform(originalPoints[i].x, shift, scale, dx);
        points[i].y = linearTransform(originalPoints[i].y, shift, scale, dy);
    }
}

void linearShiftPoints(float2 *points, int n, float shiftX, float shiftY)
{
    for(int i = 0; i < n; i++)
    {
        points[i].x = points[i].x - shiftX;
        points[i].y = points[i].y + shiftY;
    }
}

float linearScalePoints(float2 *points, int n, float scale)
{
    float maxDist = getLargestMagnitude(points, n);
    
    for(int i = 0; i < n; i++)
    {
        points[i].x = scale*points[i].x/maxDist;
        points[i].y = scale*points[i].y/maxDist;
    }
    
    return(scale/maxDist);
}
