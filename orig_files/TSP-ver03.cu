#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "./headers/dataio.h"
#include "./headers/rendering.h"
#include "./headers/drawing.h"
#include "./headers/arrays.h"

#define DIM 1024
#define BLOCK 1024


#define SLOPE_REPULSION 50.0
#define FORCE_CUTOFF 0.10
#define MAG_ATTRACTION 0.5
#define WALL_STRENGTH 200000.0
#define DAMP 20.0
#define MASS 80.0
#define BINS 8
#define MIN_BIN_DENSITY 3
#define MAX_BUBBLES 5

#define LOWER_PRESSURE_LIMIT 500
#define UPPER_PRESSURE_LIMIT 1000

#define FP "./datasets/att48/coords.txt"
#define FP_OPT "./datasets/att48/path.txt"
#define FP_LOG "./runlog.txt"

#define DRAW 1

OutputParameters params;

const int numberOfNodes = getNumberOfLines(FP);

float2 *orginalCoords = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *h_nodes = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *h_pos = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *h_vel = (float2*)malloc((numberOfNodes)*sizeof(float2));
float2 *h_acc = (float2*)malloc((numberOfNodes)*sizeof(float2));
float  *h_mass = (float*)malloc((numberOfNodes)*sizeof(float));

float h_range[BINS+1];

int h_density[BINS*BINS];
float2 h_densityCenters[BINS*BINS]; //TODO: Implement function to get the MAX_BUBBLES hightest densities
float4 h_bubbles[BINS*BINS];

int *nBodyPath = (int*)malloc((numberOfNodes)*sizeof(int));
double nBodyCost;

int *optimalPath = (int*)malloc((numberOfNodes)*sizeof(int));
double optimalCost;

double percentDiff;

float4 boundingBox;
float2 geometricCenter;
float normalizingFactor;
float outerRadius;
float innerRadius;
int outerDirection;
int innerDirection;

int mouse_x, mouse_y;
int displayFlag = 0;
int extrustionFlag = 0;
double timer;

void draw();

/* ========================================================================== */
/*                                 DATA SETUP                                 */
/* ========================================================================== */

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

void initializeBubbles(float radius)
{
    for(int i = 0; i < BINS*BINS; i++)
    {
        if(h_density[i] >= MIN_BIN_DENSITY)
        {
            h_bubbles[i].x = h_densityCenters[i].x;
            h_bubbles[i].y = h_densityCenters[i].y;
            h_bubbles[i].z = radius;
            h_bubbles[i].w = 1.0;
        }
        else
        {
            h_bubbles[i].w = 0.0;
        }
    
    }
}

void updateBubbles(float dr)
{
    for(int i = 0; i < BINS*BINS; i++)
    {
        if(h_bubbles[i].w == 1.0)
        {
            h_bubbles[i].z += dr;
        }
    }
}


double findPressureOnOuterWall(float2 *pos, float outerR, int n)
{
    int i;
	double sum, temp;
    double pi = 3.14159265358979323846;

	sum = 0.0;
	for(i = 0; i < n; i++)
	{
        temp = sqrt(pos[i].x*pos[i].x + pos[i].y*pos[i].y) - outerR;
		if( 0 < temp) 
		{
			sum += temp;
		}
    }
    double res = sum*WALL_STRENGTH/(2.0*pi*outerR);
	return(res);
}

void setNbodyInitialConditions()
{
	int i;

	for(i = 0; i < numberOfNodes; i++)
	{
		h_pos[i].x = h_nodes[i].x;
        h_pos[i].y = h_nodes[i].y;
        		
		h_vel[i].x = 0.0;
		h_vel[i].y = 0.0;
		
		h_mass[i] = MASS;
	}
}

void getNBodyPath()
{
	int i;
	double minValue;
	double *angle = (double*)malloc(numberOfNodes*sizeof(double));
    int *used = (int*)malloc(numberOfNodes*sizeof(int));
    double pi = 3.14159265358979323846;
	
	for(i = 0; i < numberOfNodes; i++)
	{
		if(h_pos[i].x == 0 && h_pos[i].y == 0)
		{
			angle[i] = 0.0;
		}
		else if(h_pos[i].x >= 0 && h_pos[i].y >= 0)
		{
			if(h_pos[i].x == 0) angle[i] = 90.0;
			else angle[i] = atan(h_pos[i].y/h_pos[i].x)*180.0/pi;
		}
		else if(h_pos[i].x < 0 && h_pos[i].y >= 0)
		{
			angle[i] = 180.0 - atan(h_pos[i].y/(-h_pos[i].x))*180.0/pi;
		}
		else if(h_pos[i].x <= 0 && h_pos[i].y < 0)
		{
			if(h_pos[i].x == 0) angle[i] = 270.0;
			else angle[i] = 180.0 + atan(h_pos[i].y/h_pos[i].x)*180.0/pi;
		}
		else
		{
			angle[i] = 360.0 - atan(-h_pos[i].y/h_pos[i].x)*180.0/pi;
		}
	}
	
	for(i = 0; i < numberOfNodes; i++)
	{
		used[i] = 0;
	}
	
	for(int k = 0; k < numberOfNodes; k++)
	{
		minValue = 400.0;
		for(i = 0; i < numberOfNodes; i++)
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

__global__ void accelerations(float2* nodes, float2* pos, float2* vel, float2* acc, float *mass, float innerRadius, float outerRadius, float4 *bubbles, int n, int b, float dt)
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
            
            if(dist <= edgeLength)
            {
                forceMag = -(edgeLength - dist)*SLOPE_REPULSION;

            }
            else if(edgeLength < dist && dist < FORCE_CUTOFF)
            {
                forceMag =  MAG_ATTRACTION/edgeLength;
            }
            else
            {
                forceMag = 0.0;
            }
            
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

double nBodyExtrustionTSP()
{
/* ------------------------------- N-body data ------------------------------ */
    float2 *d_nodes, *d_pos, *d_vel, *d_acc;
    float *d_mass;
    float dr;
    double pressure, stopSeperation;
    float normFactor;

    double t;
    float dt = 0.01;
    int drawCount;

    cudaMalloc(&d_nodes, numberOfNodes * sizeof(float2));
    cudaMalloc(&d_pos, numberOfNodes * sizeof(float2));
    cudaMalloc(&d_vel, numberOfNodes * sizeof(float2));
    cudaMalloc(&d_acc, numberOfNodes * sizeof(float2));
    cudaMalloc(&d_mass, numberOfNodes * sizeof(float));

    cudaMemcpy( d_nodes, h_nodes, numberOfNodes * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( d_pos, h_pos, numberOfNodes * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( d_vel, h_vel, numberOfNodes * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( d_acc, h_acc, numberOfNodes * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy( d_mass, h_mass, numberOfNodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

/* --------------------------- Point density data --------------------------- */
    float *d_range;
    int *d_density;
    float2 *d_densityCenters;

    int bb = BINS*BINS;

    cudaMalloc(&d_range, (BINS+1)*sizeof(float));
    cudaMemcpy(d_range, h_range, (BINS+1)*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_density, bb*sizeof(int));
    cudaMalloc(&d_densityCenters, bb*sizeof(float2));


    float4 *d_bubbles;
    cudaMalloc(&d_bubbles, bb*sizeof(float4));

    dim3 grid;
    grid.x = BINS;
    grid.y = BINS;
    grid.z = 1;
    
    getDensity<<<grid, 1>>>(d_pos, d_density, d_densityCenters, d_range, BINS, numberOfNodes);
    cudaMemcpy(h_density, d_density, bb*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_densityCenters, d_densityCenters, bb*sizeof(float2), cudaMemcpyDeviceToHost);

/* ------------------------------- Radius Data ------------------------------ */
    stopSeperation = getSmallestMagnitude(h_pos, numberOfNodes)/2.0;

    innerRadius = 0.0;
    outerRadius = getLargestMagnitude(h_pos, numberOfNodes);

    dr = outerRadius/1000;
        
    initializeBubbles(0.0);
    


    cudaMemcpy(d_bubbles, h_bubbles, bb*sizeof(float4), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    normFactor = outerRadius;
    outerRadius /= normFactor;
    innerRadius /= normFactor;
    linearScalePoints(h_pos, numberOfNodes, 1/normFactor);
/* ------------------------- Initial Draw and Pause ------------------------- */
    //printf("\n\t--- Paused for initial display. Press ENTER to continue. ---");
    
    if(DRAW) draw();
    //getchar();

/* ----------------------------- Main NBody Loop ---------------------------- */
    outerDirection = -1;
    innerDirection = 0;
    pressure = 0.0;    

    drawCount = 0;
    while(innerRadius + stopSeperation < outerRadius)
    {
        printf("\r\t☐  Running n-body extrustion.  %.0f%%", floor(100*innerRadius/outerRadius));
        outerRadius += dr*outerDirection;
        innerRadius += dr*innerDirection;
        t = 0.0;
        while(t < 2.0)
        {
            accelerations<<<1, numberOfNodes>>>(d_nodes, d_pos, d_vel, d_acc, d_mass, innerRadius, outerRadius, d_bubbles, numberOfNodes, BINS, dt);
            getDensity<<<grid, 1>>>(d_pos, d_density, d_densityCenters, d_range, BINS, numberOfNodes);
            cudaDeviceSynchronize();
            if(drawCount == 100)
            {
                if(DRAW) 
                {
                    
                    cudaMemcpy(h_density, d_density, bb*sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_densityCenters, d_densityCenters, bb*sizeof(float2), cudaMemcpyDeviceToHost);
                }
                
                cudaMemcpy(h_pos, d_pos, numberOfNodes * sizeof(float2), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_bubbles, d_bubbles, bb*sizeof(float4), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                

                updateBubbles(innerDirection*dr);
                cudaMemcpy(d_bubbles, h_bubbles, bb*sizeof(float4), cudaMemcpyHostToDevice);
                cudaDeviceSynchronize();

                normFactor = outerRadius;

                outerRadius /= normFactor;
                innerRadius /= normFactor;

                linearScalePoints(h_pos, numberOfNodes, 1/normFactor);

                if(DRAW) draw();
                drawCount = 0;
            }
            drawCount++;
            t += dt;
        }
        t = 0.0;

        float dx, dy, dist;

        for(int k = 0;k<bb;k++)
        {
            if(h_bubbles[k].w == 1.0)
            {
                dx = h_bubbles[k].x;
                dy = h_bubbles[k].y;
                dist = sqrt(dx*dx + dy*dy);
                if(dist+h_bubbles[k].z >= outerRadius)
                {
                    h_bubbles[k].w = 0.0;
                }
            }
        }
        updateBubbles(innerDirection*dr);
        cudaMemcpy(d_bubbles, h_bubbles, bb*sizeof(float4), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        

        pressure = findPressureOnOuterWall(h_pos, outerRadius, numberOfNodes);
        // Allocates storage
        char st[10];
        // Prints "Hello world!" on hello_world
        sprintf(st, "%f", pressure);
        drawText(0.5, 0.5, st);
        if(DRAW) glutSwapBuffers();

        if(pressure < LOWER_PRESSURE_LIMIT)
		{
			innerDirection = 0;
			outerDirection = -1;
		}
		else if(pressure < UPPER_PRESSURE_LIMIT)
		{
			innerDirection = 1;
            outerDirection = 0;
            initializeBubbles(innerRadius);
		}
		else
		{
			innerDirection = 0;
			outerDirection = 1;
        }
        cudaDeviceSynchronize();
    }

    
    cudaFree(d_nodes); cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_acc); cudaFree(d_mass);
    cudaFree(d_range); cudaFree(d_density); cudaFree(d_densityCenters); cudaFree(d_bubbles);
    return(0);

}

/* ========================================================================== */
/*                              OPENGL FUNCTIONS                              */
/* ========================================================================== */
void drawBubbles(float4 *bubbles, int b, float normFactor)
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

void drawDensity(int *density, float2 *densityCenters, float *range, int b)
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

void drawNBodyExtrusion(float2* pos, float innerRadius, float outerRadius, int innerDirection, int outterDirection, int n)
{

    int lineAmount = 100;
    
    float2 center = make_float2(0.0, 0.0);
    
    float innerCircleColor[] = {0.88, 0.61, 0.0};
    float outerCircleColor[] = {0.88, 0.20, 0.0};

    drawCircle(center, innerRadius, lineAmount, 2.0, innerCircleColor);
    drawCircle(center, outerRadius, lineAmount, 2.0, outerCircleColor);
    drawPoints(pos, n, 5.0, NULL);


}

void drawNBodyPath(float2 *nodes, int *path, int n)
{
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
}

void draw()
{
    drawDensity(h_density, h_densityCenters, h_range, BINS);
    drawBubbles(h_bubbles, BINS, outerRadius);
    drawNBodyExtrusion(h_pos, innerRadius, outerRadius, innerDirection, outerDirection, numberOfNodes);
    glutSwapBuffers();
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    if(displayFlag<1)
    {
        displayFlag++;
    }

    //drawDensity(nodes, numberOfNodes, b, scale);
    if(extrustionFlag == 0)
    {
        
        nBodyExtrustionTSP();
        printf("\r\t🗹  N-body extrustion completed.\n");

        getNBodyPath();
        printf("\t🗹  N-body path obtained.\n");
        drawNBodyPath(h_nodes, nBodyPath, numberOfNodes);
 
        nBodyCost = getPathCost(orginalCoords, nBodyPath, numberOfNodes, 0);
        printf("\t🗹  N-body path cost calculated. Cost = %f\n", nBodyCost);

        percentDiff = 100*(nBodyCost - optimalCost)/optimalCost;
        printf("\t🗹  N-body cost percent difference calculated. Percent Difference = %f%%\n", percentDiff);
        endTimer(&timer);

        params.slopeRepulsion = SLOPE_REPULSION;
        params.forceCutoff = FORCE_CUTOFF;
        params.magAttraction = MAG_ATTRACTION;
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
        logRun(FP_LOG, params);
        printf("\t🗹  Logged run parameters and results to %s\n", FP_LOG);

        extrustionFlag = 1;
    }
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
    startTimer(&timer);
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

    linspace(h_range, -1.0, 1.0, BINS+1, 1);
    linearTransformPoints(orginalCoords, h_nodes, numberOfNodes, -0.5, 1.0, dx, dy);
    printf("\t🗹  Transformed node positions.\n");

    geometricCenter = getGeometricCenter(h_nodes, numberOfNodes);
    printf("\t🗹  Got geometric center.\n");

	linearShiftPoints(h_nodes, numberOfNodes, geometricCenter.x, geometricCenter.y);
	geometricCenter = getGeometricCenter(h_nodes, numberOfNodes);
	printf("\t🗹  Aligned geometric center to origin.\n");

    normalizingFactor = linearScalePoints(h_nodes, numberOfNodes, 1.0);
	printf("\t🗹  Normalized nodes. Normalizing factor = %f\n", normalizingFactor);

    setNbodyInitialConditions();
    printf("\t🗹  Set n-body initial conditions.");

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
        display();
    }

    /* ------------------------------ OpenGL Calls ------------------------------ */

    free(h_nodes); free(h_pos); free(h_vel); free(h_acc); free(h_mass);
    free(nBodyPath); free(optimalPath);
}