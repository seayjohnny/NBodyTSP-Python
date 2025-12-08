#pragma once

#include "lintrans.cuh"

struct Node 
{
    int id;
    float2 initpos;
    float2 pos;
    float2 vel;
    float2 acc;
    float mass;
};

/* ---------------- getters and setters for an array of nodes --------------- */
void getNodeInitPositions(float2 *pos, Node* nodes, int n){ for(int i=0;i<n;i++) pos[i] = nodes[i].initpos; }

void getNodePositions(float2 *pos, Node* nodes, int n){ for(int i=0;i<n;i++) pos[i] = nodes[i].pos; }
void setNodePositions(Node* nodes, float2 *pos, int n){ for(int i=0;i<n;i++) nodes[i].pos = pos[i]; }

void getNodeVelocities(float2 *vel, Node* nodes, int n){ for(int i=0;i<n;i++) vel[i] = nodes[i].vel; }
void setNodeVelocities(Node* nodes, float2 *vel, int n){ for(int i=0;i<n;i++) nodes[i].vel = vel[i]; }

void getNodeAccelerations(float2 *acc, Node* nodes, int n){ for(int i=0;i<n;i++) acc[i] = nodes[i].acc; }
void setNodeAccelerations(Node* nodes, float2 *acc, int n){ for(int i=0;i<n;i++) nodes[i].acc = acc[i]; }

void getNodeMass(float *mass, Node* nodes, int n){  *mass = nodes[0].mass; }
void setNodeMass(Node* nodes, float mass, int n){ for(int i=0;i<n;i++) nodes[i].mass = mass; }

/* ------------------- transfomations on an array of nodes ------------------ */
void shiftNodes(Node* nodes, int n, float shiftX, float shiftY){ for(int i = 0;i < n;i++) linearShiftPoint(&nodes[i].pos, shiftX, shiftY); }
void shiftNodes(Node* nodes, int n, float2 shift){ for(int i = 0;i < n;i++) linearShiftPoint(&nodes[i].pos, shift); }
void scaleNodes(Node* nodes, int n, float scaleX, float scaleY){ for(int i = 0;i < n;i++) linearScalePoint(&nodes[i].pos, scaleX, scaleY); }
void scaleNodes(Node* nodes, int n, float2 scale){ for(int i = 0;i < n;i++) linearScalePoint(&nodes[i].pos, scale); }
void scaleNodes(Node* nodes, int n, float scale){ for(int i = 0;i < n;i++) linearScalePoint(&nodes[i].pos, scale, scale); }

float2 minPositionValues(Node* nodes, int n)
{
    float2 minValues = nodes[0].pos;
    for(int i = 1; i < n; i++)
    {
        if(nodes[i].pos.x < minValues.x) minValues.x = nodes[i].pos.x;
        if(nodes[i].pos.y < minValues.y) minValues.y = nodes[i].pos.y;
    }

    return minValues;
}

float2 maxPositionValues(Node* nodes, int n)
{
    float2 maxValues = nodes[0].pos;
    for(int i = 1; i < n; i++)
    {
        if(nodes[i].pos.x > maxValues.x) maxValues.x = nodes[i].pos.x;
        if(nodes[i].pos.y > maxValues.y) maxValues.y = nodes[i].pos.y;
    }

    return maxValues;
}

float4 getBoundingBox(Node* nodes, int n)
{
    float2 min = minPositionValues(nodes, n);
    float2 max = maxPositionValues(nodes, n);

	return make_float4(min.x, min.y, max.x, max.y);
}

float2 getGeometricCenter(Node* nodes, int n)
{
    float2 geometricCenter = {0, 0};
    for(int i = 0; i < n; i++) geometricCenter += nodes[i].pos;
    geometricCenter /= n;
    return geometricCenter;
}


int getFarthestNodeId(Node* nodes, int n)
{
    int id;
    float dist = 0.0;
    for(int i = 0;i < n;i++)
    {
        float m = mag(nodes[i].pos);
        if(m > dist)
        {
            dist = m;
            id = i;
        }
    }

    return id;
}

float getFarthestNodeDist(Node* nodes, int n)
{
    float dist = 0.0;
    for(int i = 0;i < n;i++)
    {
        float m = mag(nodes[i].pos);
        if(m > dist)
        {
            dist = m;
        }
    }

    return dist;
}

void normalizeNodePositions(Node* nodes, int n)
{
    float m = getFarthestNodeDist(nodes, n);
    for(int i = 0;i < n;i++)
    {
        nodes[i].pos /= m;
    }

}

void resetNodeInitPositions(Node* nodes, int n)
{
    for(int i = 0;i < n;i++)
    {
        nodes[i].initpos = nodes[i].pos;
    }
}

void getNBodyPath(int* path, Node* nodes, int n)
{
	int i;
	double minValue;
	double *angle = (double*)malloc(n*sizeof(double));
    int *used = (int*)malloc(n*sizeof(int));
    double pi = 3.14159265358979323846;
	
	for(i = 0; i < n; i++)
	{
		if(nodes[i].pos.x == 0 && nodes[i].pos.y == 0)
		{
			angle[i] = 0.0;
		}
		else if(nodes[i].pos.x >= 0 && nodes[i].pos.y >= 0)
		{
			if(nodes[i].pos.x == 0) angle[i] = 90.0;
			else angle[i] = atan(nodes[i].pos.y/nodes[i].pos.x)*180.0/pi;
		}
		else if(nodes[i].pos.x < 0 && nodes[i].pos.y >= 0)
		{
			angle[i] = 180.0 - atan(nodes[i].pos.y/(-nodes[i].pos.x))*180.0/pi;
		}
		else if(nodes[i].pos.x <= 0 && nodes[i].pos.y < 0)
		{
			if(nodes[i].pos.x == 0) angle[i] = 270.0;
			else angle[i] = 180.0 + atan(nodes[i].pos.y/nodes[i].pos.x)*180.0/pi;
		}
		else
		{
			angle[i] = 360.0 - atan(-nodes[i].pos.y/nodes[i].pos.x)*180.0/pi;
		}
	}
	
	for(i = 0; i < n; i++)
	{
		used[i] = 0;
	}
	
	for(int k = 0; k < n; k++)
	{
		minValue = 400.0;
		for(i = 0; i < n; i++)
		{
			if(angle[i] < minValue && used[i] == 0)
			{
				minValue = angle[i];
				path[k] = i;
			}
		}
		used[path[k]] = 1;
		//printf("path[%d] = %d\n", k, path[k]);
    }
    free(angle);
    free(used);
}