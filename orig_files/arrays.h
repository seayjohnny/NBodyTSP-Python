#ifndef ARRAYSH
#define ARRAYSH

#include <math.h>

void linspace(float* vec, float start, float stop, int num, int useEndpoint)
{
/*	Create 'num' evenly spaced samples, calculated over the interval ['start', 'stop'].
 *	The endpoint of the interval can be excluded.
 */

	int q = 0;	
	float div;
	float delta;	
	float step;
		
	if(num > 1)
	{
		if(useEndpoint == 1)
		{
			q = 1;
		}

		div = num - q;
		delta = stop - start;
		step = float(delta)/div;

		for(int i = 0;i<num;i++)
		{
			vec[i] = i*step + start;
		}		
	}
}

float getLargestMagnitude(float2* vec, int n)
{
	float maxMag = 0.0;
    for(int i = 0; i < n; i++)
    {
        float mag = sqrtf( (vec[i].x*vec[i].x) + (vec[i].y*vec[i].y));
        if(mag > maxMag) maxMag = mag;
    }
	
	return(maxMag);
}

float getSmallestMagnitude(float2* vec, int n)
{
	float minMag = sqrtf( (vec[0].x*vec[0].x) + (vec[0].y*vec[0].y));
    for(int i = 1; i < n; i++)
    {
        float mag = sqrtf( (vec[i].x*vec[i].x) + (vec[i].y*vec[i].y));
        if(mag < minMag) minMag = mag;
    }
	
	return(minMag);
}

#endif