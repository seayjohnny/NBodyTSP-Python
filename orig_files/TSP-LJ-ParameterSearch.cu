#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "./headers/dataio.h"
#include "./headers/rendering.h"
#include "./headers/drawing.h"
#include "./headers/arrays.h"

#define DIM 1024
#define BLOCK 1024


/* #define M -0.0015
#define P 5.0
#define Q 11.0 */
#define WALL_STRENGTH 200000.0
#define DAMP 200.0
#define MASS 80.0
#define BINS 1
#define MIN_BIN_DENSITY 0
#define MAX_BUBBLES 5

#define LOWER_PRESSURE_LIMIT 500
#define UPPER_PRESSURE_LIMIT 5000

#define FP "./datasets/ch150/coords.txt"
#define FP_OPT "./datasets/ch150/path.txt"
#define FP_LOG "./runlog.txt"

#define DRAW 1

OutputParametersLJ params;

int numberOfNodes = getNumberOfLines(FP);
float2 *orginalCoords = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *nodes = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *posNbody = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *velNbody = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *accNbody = (float2*)malloc((numberOfNodes)*sizeof(float2));
float  *massNbody = (float*)malloc((numberOfNodes)*sizeof(float));
float4 *bubbles = (float4*)malloc(BINS*BINS*sizeof(float4));

int *nBodyPath = (int*)malloc((numberOfNodes)*sizeof(int));
double nBodyCost;

int *optimalPath = (int*)malloc((numberOfNodes)*sizeof(int));
double optimalCost;

double percentDiff;

float4 boundingBox;
float2 geometricCenter;
float normalizingFactor;

int mouse_x, mouse_y;
int displayFlag = 0;
int extrustionFlag = 0;
double timer;

void drawNBodyExtrusion(float2* pos, float innerRadius, float outerRadius, int innerDirection, int outterDirection, int n);
void drawDensity(int *density, float2 *densityCenters, float *range, int b);
void drawBubbles(float4 *bubbles, int b, float normFactor);

/* ========================================================================== */
/*                                 DATA SETUP                                 */
/* ========================================================================== */

float randInRange(float a, float b)
{
    return a + rand()*(b - a)/((float)RAND_MAX);
}

float2 maxValues(float2 *nodes, int n)
{
    float maxX = nodes[0].x;
    float maxY = nodes[0].y;

    for(int i = 1; i < n; i++)
    {
        if(nodes[i].x > maxX) maxX = nodes[i].x;
        if(nodes[i].y > maxY) maxY = nodes[i].y;
    }
    return(make_float2(maxX, maxY));
}

float2 minValues(float2 *nodes, int n)
{
    float minX = nodes[0].x;
    float minY = nodes[0].y;

    for(int i = 1; i < n; i++)
    {
        if(nodes[i].x < minX) minX = nodes[i].x;
        if(nodes[i].y < minY) minY = nodes[i].y;
    }

    return(make_float2(minX, minY));
}

float2 getGeometricCenter(float2 *nodes, int n)
{
    float2 geometricCenter;
	
	geometricCenter.x = 0.0;
	geometricCenter.y = 0.0;

	for(int i = 0; i < n; i++)
	{
		geometricCenter.x += nodes[i].x;
		geometricCenter.y += nodes[i].y;
	}
	
	geometricCenter.x /= (float)n;
	geometricCenter.y /= (float)n;

	return(geometricCenter);
}

float4 getBoundingBox(float2 *nodes, int n)
{
    float2 min = minValues(nodes, n);
    float2 max = maxValues(nodes, n);

	return(make_float4(min.x, min.y, max.x, max.y));
}

void initializeBubbles(float4 *bubbles, int *density, float2* densityCenters, float radius, int b)
{
    for(int i = 0; i < b*b; i++)
    {
        if(density[i] >= MIN_BIN_DENSITY && MIN_BIN_DENSITY != 0)
        {
            bubbles[i].x = densityCenters[i].x;
            bubbles[i].y = densityCenters[i].y;
            bubbles[i].z = radius;
            bubbles[i].w = 1.0;
        }
        else
        {
            bubbles[i].w = 0.0;
        }
    
    }
}

void updateBubbles(float4 *bubbles, float dr, int b)
{
    for(int i = 0; i < b*b; i++)
    {
        if(bubbles[i].w == 1.0)
        {
            bubbles[i].z += dr;
        }
    }
}

double findPressureOnOuterWall(float2 *pos, float outerRadius, int n)
{
    int i;
	double sum, temp;
    double pi = 3.14159265358979323846;

	sum = 0.0;
	for(i = 0; i < n; i++)
	{
        temp = sqrt(pos[i].x*pos[i].x + pos[i].y*pos[i].y) - outerRadius;
		if( 0 < temp) 
		{
			sum += temp;
		}
	}
	return(sum*WALL_STRENGTH/(2.0*pi*outerRadius));
}

void setNbodyInitialConditions(float2 *node, float2 *pos, float2 *vel, float* mass, int n)
{
	int i;

	for(i = 0; i < n; i++)
	{
		pos[i].x = node[i].x;
		pos[i].y = node[i].y;
		
		vel[i].x = 0.0;
		vel[i].y = 0.0;
		
		mass[i] = MASS;
	}
}

void getNBodyPath(float2 *pos, int* path, int n)
{
	int i;
	double minValue;
	double *angle = (double*)malloc(n*sizeof(double));
    int *used = (int*)malloc(n*sizeof(int));
    double pi = 3.14159265358979323846;
	
	for(i = 0; i < n; i++)
	{
		if(posNbody[i].x == 0 && posNbody[i].y == 0)
		{
			angle[i] = 0.0;
		}
		else if(posNbody[i].x >= 0 && posNbody[i].y >= 0)
		{
			if(posNbody[i].x == 0) angle[i] = 90.0;
			else angle[i] = atan(posNbody[i].y/posNbody[i].x)*180.0/pi;
		}
		else if(posNbody[i].x < 0 && posNbody[i].y >= 0)
		{
			angle[i] = 180.0 - atan(posNbody[i].y/(-posNbody[i].x))*180.0/pi;
		}
		else if(posNbody[i].x <= 0 && posNbody[i].y < 0)
		{
			if(posNbody[i].x == 0) angle[i] = 270.0;
			else angle[i] = 180.0 + atan(posNbody[i].y/posNbody[i].x)*180.0/pi;
		}
		else
		{
			angle[i] = 360.0 - atan(-posNbody[i].y/posNbody[i].x)*180.0/pi;
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
				nBodyPath[k] = i;
			}
		}
		used[nBodyPath[k]] = 1;
		//printf("path[%d] = %d\n", k, path[k]);
    }
    free(angle);
    free(used);
}

double getPathCost(float2 *node, int *path, int n, int debug)
{
    double cost;
    float dx, dy, temp;

    for(int i = 0; i < n; i++)
    {
        dx = node[path[i%n]].x - node[path[(i+1)%n]].x;
        dy = node[path[i%n]].y - node[path[(i+1)%n]].y;
        temp = sqrt(dx*dx + dy*dy);
        if(debug)
        {
            printf("\t%d (%f, %f) -> %d (%f, %f) = %f\n", path[i%n], node[path[i%n]].x, node[path[i%n]].y, path[(i+1)%n], node[path[(i+1)%n]].x, node[path[(i+1)%n]].y, temp);
            //printf("\t%f\n",temp);

        }
        cost += sqrt(dx*dx + dy*dy);
    }

    return(cost);
}

/* ========================================================================== */
/*                               CUDA FUNCTIONS                               */
/* ========================================================================== */

__global__ void accelerations(float2* nodes, float2* pos, float2* vel, float2* acc, float *mass, float innerRadius, float outerRadius, float4 *bubbles, int n, int b, float dt, float M, float p, float q)
{
    float2 force;
    float2 node, nodePos;
    float dx, dy, dist, edx, edy, edgeLength;
    float radius, forceMag;
    __shared__ float2 shNodes[BLOCK], shPos[BLOCK];

    int id = threadIdx.x;

    force.x = 0.0;
    force.y = 0.0;

    node.x = nodes[id].x;
	node.y = nodes[id].y;
	nodePos.x = pos[id].x;
	nodePos.y = pos[id].y;
		    
    shPos[id] = nodePos;
    shNodes[id] = node;
    __syncthreads();

    for(int i = 0; i < n; i++)	
    {
        if(i != id) 
        {
            dx = shPos[i].x - nodePos.x;
            dy = shPos[i].y - nodePos.y;
            dist = sqrtf(dx*dx + dy*dy);
            
            edx = shNodes[i].x - node.x;
            edy = shNodes[i].y - node.y;
            edgeLength = sqrtf(edx*edx + edy*edy);
            float a = powf(q/p, 1/(q-p));
            float b = powf(a*edgeLength, p);
            double H = M*b/(1 - p/q);
            float c = powf(edgeLength/dist, q-p);

            forceMag = (c-1)*H/powf(dist, p);
            
            force.x += forceMag*dx/dist;
            force.y += forceMag*dy/dist;
        }
    }
    
	if(id < n)
	{
		// Forces between node and the walls
		dx = nodePos.x;
		dy = nodePos.y; 
		radius = sqrtf(dx*dx + dy*dy);
	
		if(radius < innerRadius) // Inside inner wall
		{
            
			forceMag = WALL_STRENGTH*(innerRadius - radius);
			force.x += forceMag*dx/radius;
			force.y += forceMag*dy/radius;
		}
		else if(radius > outerRadius) // Outside outer wall
		{
			forceMag = WALL_STRENGTH*(outerRadius - radius);
			force.x += forceMag*dx/radius;
			force.y += forceMag*dy/radius;
		}
        
        if(id < n)
        {
            // Forces between node and bubbles
            for(int k=0;k<b*b;k++)
            {
                if(bubbles[k].z > 0.0)
                {
                    dx = bubbles[k].x - nodePos.x;
                    dy = bubbles[k].y - nodePos.y;
                    dist = sqrt(dx*dx + dy*dy);

                    if(dist < bubbles[k].z)
                    {
                        forceMag = -WALL_STRENGTH*(bubbles[k].z - dist);
                        force.x += forceMag*dx/radius;
                        force.y += forceMag*dy/radius;
                    }
                }
            }
        }
		// Adding on damping force.
		force.x += -DAMP*vel[id].x;
		force.y += -DAMP*vel[id].y;
		
		// Creating the accelerations.
	    acc[id].x = force.x/mass[id];
	    acc[id].y = force.y/mass[id];
    }

    

    __syncthreads();

    if(id < n)
    {
	    vel[id].x += acc[id].x*dt;
		vel[id].y += acc[id].y*dt;
		
		pos[id].x  += vel[id].x*dt;
		pos[id].y  += vel[id].y*dt;
    }

}

__global__ void getDensity(float2 *points, int *density, float2 *densityCenters, float *range, int b, int n)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int id = x + y*gridDim.x*blockDim.x;

    float xr[2] = {range[blockIdx.x], range[blockIdx.x+1]};
    float yr[2] = {range[b-1-blockIdx.y], range[b-1-blockIdx.y+1]};
    float xBar = 0.0;
    float yBar = 0.0;
    for(int i = 0; i < n; i++)
    {
        density[id] = 0;
    }
    __syncthreads();

    for(int i = 0; i < n; i++)
    {
        if(points[i].x >= xr[0] && points[i].x <= xr[1])
        {
            if(points[i].y >= yr[0] && points[i].y <= yr[1])
            {
                density[id] += 1;
                xBar += points[i].x;
                yBar += points[i].y;
            }
        }
    }
    if(density[id])
    {
        densityCenters[id].x = xBar/density[id];
        densityCenters[id].y = yBar/density[id];
    }

    __syncthreads();

}

double nBodyExtrustionTSP(float2* nodes, float2* pos, float2* vel, float2* acc, float *mass, int n, int b, float M, float p, float q)
{

/* ------------------------------- N-body data ------------------------------ */
    float2 *nodesGPU, *posGPU, *velGPU, *accGPU;
    float *massGPU;
    float innerRadius, outerRadius, dr;
    double pressure, stopSeperation;
    int innerWallDirection, outerWallDirection;
    double t;
    float dt = 0.01;
    int drawCount;

    cudaMalloc(&nodesGPU, n * sizeof(float2));
    cudaMalloc(&posGPU, n * sizeof(float2));
    cudaMalloc(&velGPU, n * sizeof(float2));
    cudaMalloc(&accGPU, n * sizeof(float2));
    cudaMalloc(&massGPU, n * sizeof(float));

    cudaMemcpy( nodesGPU, nodes, n * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( posGPU, pos, n * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( velGPU, vel, n * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( accGPU, acc, n * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( massGPU, mass, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

/* --------------------------- Point density data --------------------------- */
    float *range = (float*)malloc((b+1)*sizeof(float));
    float *rangeGPU;
    int *density = (int*)malloc((b*b)*sizeof(int));
    int *densityGPU;
    float2 *densityCenters = (float2*)malloc(b*b*sizeof(float2));
    float2 *densityCentersGPU;

    linspace(range, -1.0, 1.0, b+1, 1);
    cudaMalloc(&rangeGPU, (b+1)*sizeof(float));
    cudaMemcpy(rangeGPU, range, (b+1)*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&densityGPU, b*b*sizeof(int));
    cudaMalloc(&densityCentersGPU, b*b*sizeof(float2));

/* ------------------------------- Bubble data ------------------------------ */
    float4 *bubblesGPU;
    cudaMalloc(&bubblesGPU, b*b*sizeof(float4));

    dim3 grid;
    grid.x = b;
    grid.y = b;
    grid.z = 1;
    
    getDensity<<<grid, 1>>>(posGPU, densityGPU, densityCentersGPU, rangeGPU, b, n);
    cudaMemcpy(density, densityGPU, b*b*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(densityCenters, densityCentersGPU, b*b*sizeof(float2), cudaMemcpyDeviceToHost);
    
    stopSeperation = getSmallestMagnitude(pos, n)/2.0;
    innerRadius = 0.0;
    outerRadius = getLargestMagnitude(pos, n);

    dr = outerRadius/1000;

    initializeBubbles(bubbles, density, densityCenters, innerRadius, b);
    cudaMemcpy(bubblesGPU, bubbles, b*b*sizeof(float4), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    drawDensity(density, densityCenters, range, b);
    drawBubbles(bubbles, b, outerRadius);
    drawNBodyExtrusion(pos, innerRadius, outerRadius, 0, 0, n);
    //printf("\n\t--- Paused for initial display. Press ENTER to continue. ---");
    if(DRAW) glutSwapBuffers();
    //getchar();
    
    outerWallDirection = -1;
    innerWallDirection = 0;
    pressure = 0.0;
    
    drawCount = 0;  
    while(innerRadius + stopSeperation < outerRadius)
    {
        printf("\r\t☐  Running n-body extrustion.  %.0f%%", floor(100*innerRadius/outerRadius));
        outerRadius += dr*outerWallDirection;
        innerRadius += dr*innerWallDirection;
        t = 0.0;
        while(t < 2.0)
        {
            accelerations<<<1, numberOfNodes>>>(nodesGPU, posGPU, velGPU, accGPU, massGPU, innerRadius, outerRadius, bubblesGPU, n, b, dt, M, p, q);
            getDensity<<<grid, 1>>>(posGPU, densityGPU, densityCentersGPU, rangeGPU, b, n);
            cudaDeviceSynchronize();
            if(drawCount == 100)
            {
                if(DRAW)
                {
                    
                    cudaMemcpy(density, densityGPU, b*b*sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(densityCenters, densityCentersGPU, b*b*sizeof(float2), cudaMemcpyDeviceToHost);
                }
                
                cudaMemcpy(pos, posGPU, n * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaMemcpy(bubbles, bubblesGPU, b*b*sizeof(float4), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                
                updateBubbles(bubbles, innerWallDirection*dr, b);
                cudaMemcpy(bubblesGPU, bubbles, b*b*sizeof(float4), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();


                drawDensity(density, densityCenters, range, b);
                drawBubbles(bubbles, b, outerRadius);
                drawNBodyExtrusion(pos, innerRadius, outerRadius, innerWallDirection, outerWallDirection, n);
                if(DRAW) glutSwapBuffers();

                drawCount = 0;
            }
            drawCount++;
            t += dt;
        }

        float dx, dy, dist;

        for(int k = 0;k<b*b;k++)
        {
            if(bubbles[k].w == 1.0)
            {
                dx = bubbles[k].x;
                dy = bubbles[k].y;
                dist = sqrt(dx*dx + dy*dy);
                if(dist+bubbles[k].z >= outerRadius)
                {
                    bubbles[k].w = 0.0;
                }
            }
        }
        updateBubbles(bubbles, innerWallDirection*dr, b);
        cudaMemcpy(bubblesGPU, bubbles, b*b*sizeof(float4), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();


        pressure = findPressureOnOuterWall(pos, outerRadius, n);

        if(pressure < LOWER_PRESSURE_LIMIT)
		{
			innerWallDirection = 0;
			outerWallDirection = -1;
		}
		else if(pressure < UPPER_PRESSURE_LIMIT)
		{
			innerWallDirection = 1;
            outerWallDirection = 0;
            initializeBubbles(bubbles, density, densityCenters, innerRadius, b);
		}
		else
		{
			innerWallDirection = 0;
			outerWallDirection = 1;
        }
        cudaDeviceSynchronize();
    }

    
    cudaFree(nodesGPU); cudaFree(posGPU); cudaFree(velGPU); cudaFree(accGPU); cudaFree(massGPU);
    cudaFree(rangeGPU); cudaFree(densityGPU); cudaFree(densityCentersGPU); cudaFree(bubblesGPU);
    cudaDeviceSynchronize();
    free(range); free(density); free(densityCenters); free(bubbles);
    return(0);
}

/* ========================================================================== */
/*                              OPENGL FUNCTIONS                              */
/* ========================================================================== */
void drawBubbles(float4 *bubbles, int b, float normFactor)
{
    if(DRAW)
    {
        float2 center;
        float color[] = {0.2, 0.8, 1.0};
        for(int i = 0; i < b*b;i++)
        {
            if(bubbles[i].z > 0.0)
            {
                center = make_float2(bubbles[i].x, bubbles[i].y);
                drawCircle(center, bubbles[i].z/normFactor, 50, 1.0, color);
            }
        }
    }

}

void drawDensity(int *density, float2 *densityCenters, float *range, int b)
{
    if(DRAW)
    {
        glClear(GL_COLOR_BUFFER_BIT);
        for(int i = 0; i < b*b; i++)
        {
            if(density[i])
            {
                float xr[2] = {range[i%b], range[i%b+1]};
                float yr[2] = {range[b-1-i/b], range[b-1-i/b+1]};
    
                drawRect(make_float2(xr[0], yr[0]), make_float2(xr[1]-xr[0], yr[1]-yr[0]), ((float)density[i])/sqrtf(numberOfNodes));
    
            }
        }
        //linearScalePoints(centers_cpu, b*b, 1.5);
        for(int i = 0; i < b*b; i++)
        {
            if(density[i])
            {
                float color[] = {0.3, 0.3, 1.0}; 
                drawPoint(densityCenters[i], 2.0, color);
            }
        }
    }

}

void drawNBodyExtrusion(float2* pos, float innerRadius, float outerRadius, int innerDirection, int outterDirection, int n)
{

    int lineAmount = 100;
    
    float2 center = make_float2(0.0, 0.0);

    
    float innerCircleColor[] = {0.88, 0.61, 0.0};
    float outerCircleColor[] = {0.88, 0.20, 0.0};

    if(DRAW){
        drawCircle(center, innerRadius, lineAmount, 2.0, innerCircleColor);
        drawCircle(center, outerRadius, lineAmount, 2.0, outerCircleColor);
        drawPoints(pos, n, 5.0, NULL);
    }

}

void drawNBodyPath(float2 *nodes, int *path, int n)
{
    if(DRAW){
        glClear(GL_COLOR_BUFFER_BIT);

        glLineWidth(2.0);
        glColor3f(1.0, 0.2, 0.2);
        glBegin(GL_LINE_LOOP);
        for(int i = 0; i < n; i++)
        {
            glVertex2f(nodes[path[i]].x, nodes[path[i]].y);
        }
        glEnd();


        drawPoints(nodes, n, 5.0, NULL);
        glutSwapBuffers();
    }
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    if(displayFlag<1)
    {
        displayFlag++;
    }
    int b = BINS;
    float scale = 1.0;
    float colorNodes[3] = {1.0, 0.1, 0.1};
    float colorCenter[3] = {0.0, 1.0, 0.2};

    float M, p, q;
    M = randInRange(-0.0025, -0.0005);
    p = randInRange(3.0, 13.0);
    q = p + randInRange(1.5, 3.0);

    //drawDensity(nodes, numberOfNodes, b, scale);
  
    startTimer(&timer);
    setNbodyInitialConditions(nodes, posNbody, velNbody, massNbody, numberOfNodes);
    printf("\t🗹  Set n-body initial conditions.\n");

    drawPoints(nodes, numberOfNodes, 2.5, colorNodes);
    drawGrid(2*scale/b, 2*scale/b, scale);
    drawPoint(geometricCenter, 2.5, colorCenter);


    nBodyExtrustionTSP(nodes, posNbody, velNbody, accNbody, massNbody, numberOfNodes, b, M, p, q);
    printf("\r\t🗹  N-body extrustion completed.\n");
    getNBodyPath(posNbody, nBodyPath, numberOfNodes);
    
    if(DRAW) drawNBodyPath(nodes, nBodyPath, numberOfNodes);

    printf("\t🗹  N-body path obtained.\n");
    nBodyCost = getPathCost(orginalCoords, nBodyPath, numberOfNodes, 0);
    printf("\t🗹  N-body path cost calculated. Cost = %f\n", nBodyCost);
    percentDiff = 100*(nBodyCost - optimalCost)/optimalCost;
    printf("\t🗹  N-body cost percent difference calculated. Percent Difference = %f%%\n", percentDiff);
    endTimer(&timer);
    params.m = M;
    params.p = p;
    params.q = q;
    params.wallStrength = WALL_STRENGTH;
    params.damp = DAMP;
    params.mass = MASS;
    params.bins = BINS;
    params.minBinDensity = MIN_BIN_DENSITY;
    params.lowerPressureLimit = LOWER_PRESSURE_LIMIT;
    params.upperPressureLimit = UPPER_PRESSURE_LIMIT;
    params.optimalCost = optimalCost;
    params.nBodyCost = nBodyCost;
    params.percentDiff = percentDiff;
    params.drawn = DRAW;
    params.runTime = timer;
    logLJRunShort(FP_LOG, params);
    printf("\t🗹  Logged run parameters and results to %s\n", FP_LOG);

    extrustionFlag = 1;
    
}

void mouseMotion(int mx, int my)
{
    mouse_x = mx;
    mouse_y = my;
    //glutPostRedisplay();
}

/* ========================================================================== */
/*                                    MAIN                                    */
/* ========================================================================== */

int main(int argc, char** argv)
{
    
    setbuf(stdout, NULL);
    /* ----------------------- Loading and Setting up Data ---------------------- */
    loadOptimalPath(FP_OPT, optimalPath, numberOfNodes);
    loadNodesFromFile(FP, orginalCoords);
    printf("\t🗹  Loaded node positions and optimal path.\n");

    optimalCost = getPathCost(orginalCoords, optimalPath, numberOfNodes, 0);
    printf("\t🗹  Obtained optimal cost. Cost = %f\n", optimalCost);

    boundingBox = getBoundingBox(orginalCoords, numberOfNodes);
    printf("\t🗹  Got bounding box.\n");

    float dx = boundingBox.z - boundingBox.x;
    float dy = boundingBox.w - boundingBox.y;


    linearTransformPoints(orginalCoords, nodes, numberOfNodes, -0.5, 1.0, dx, dy);
    printf("\t🗹  Transformed node positions.\n");

    geometricCenter = getGeometricCenter(nodes, numberOfNodes);
    printf("\t🗹  Got geometric center.\n");

	linearShiftPoints(nodes, numberOfNodes, geometricCenter.x, geometricCenter.y);
	geometricCenter = getGeometricCenter(nodes, numberOfNodes);
	printf("\t🗹  Aligned geometric center to origin.\n");

    normalizingFactor = linearScalePoints(nodes, numberOfNodes, 1.0);
	printf("\t🗹  Normalized nodes. Normalizing factor = %f\n", normalizingFactor);

    setNbodyInitialConditions(nodes, posNbody, velNbody, massNbody, numberOfNodes);
    printf("\t🗹  Set n-body initial conditions.\n");

    if(DRAW)
    {
        glutInit(&argc,argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_MULTISAMPLE);
        glutInitWindowSize(DIM,DIM);
        //glutInitWindowPosition(2800,50);
        glutCreateWindow("Traveling Salesman Problem");
        glutDisplayFunc(display);
        glutPassiveMotionFunc(mouseMotion);
        glClearColor(0.1, 0.1, 0.1, 0.1);
        glEnable(GL_MULTISAMPLE_ARB);
        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_BLEND);
        glutMainLoop();
    }
    else
    {
        for(int i = 0; i < 3; i++)
        {
            srand(time(0));
            display();
        }
    }

    printf(">>>>>>>>>>>>>>>>>>>>>>>> DONE");
    free(nodes); free(posNbody); free(velNbody); free(accNbody); free(massNbody);
    free(nBodyPath); free(optimalPath);
}