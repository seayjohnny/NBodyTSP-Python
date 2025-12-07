//Maybe add a random pertibation
//I don't like that the wall strength is just an arbitrary large number
//work on pointer in exact
//Move the point off (0,0) before running the extrution program
//Move wall by a set number because the wall will not be a fixed distance apart
//
// Clean up extrution
// worhs great up to 100 
// Think about setting atraction as gravity and repultion as linear. WIll need epsilon to remove sengularity
//nvcc TravelingSalesManComparison12-9-18.cu -o TSPCompare12918 -lglut -lGL -lm
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "./headers/dataio.h"

#define PI 3.14159265359
#define BIG_NUMBER 1000000.0
#define SMALL_NUMBER 0.0000001

#define BLOCK 256

#define X_WINDOW 800
#define Y_WINDOW 800

#define X_MAX (1.1)
#define X_MIN (-1.1)

#define Y_MAX (1.1)
#define Y_MIN (-1.1)

#define RANDOM_FILL_SHAPE_CIRCLE -1
#define RANDOM_FILL_SHAPE_SQUARE 1

#define TIME_STEP_SIZE 0.002
#define STEPS_BETWEEN_VIEWING 100
#define STARTING_POINT_FOR_NUMBER_OF_MOVES 1000
#define RADIUS_STEP_SIZE 0.001
#define TIME_BETWEEN_WALL_MOVES 1.0

#define LOWER_PRESSURE_LIMIT 500
#define UPPER_PRESSURE_LIMIT 1000

#define RELAX_TIME 10.0

#define WALL_STRENGTH 200000.0

#define SLOPE_REPULSION 100.0
#define	MAG_ATRACTION 10.0
#define FORCE_CUTOFF 10.0

#define DAMP 20.0 
#define MASS 80.0

#define NODES_TOO_CLOSE 0.000001

#define DRAW_EXHAUSTIVE_PATH 1
#define DRAW_NEAREST_NEIGHBOR_PATH -1
#define DRAW_NBODY_EXTRUSION_PATH 1
#define DELAY_TO_RECORD -1
#define PRINT_EDGE_COST -1
#define PRINT_EXHAUSTIVE_PATHS -1
#define PRINT_PATHS 1
#define PRINT_RAW_DATA_FILE 1


// Globals
FILE *RawDataFile;
FILE *StatsFile;
int clicked = 0;

//Function prototypes
float x_machine_to_x_screen(int x);
float y_machine_to_y_screen(int y);
float x_machine_to_x_world(int x);
float y_machine_to_y_world(int y);
float x_world_to_x_screen(float x);
float y_world_to_y_screen(float y);
void openRawDataFile(int scope, int numberOfRuns);
void placeNodesRandom(float4 *node, unsigned int srandSeed, int scope, int n);
int checkNodes(float4 *node, int n);
void placeNodesGrid(float4 *node, int rows, int columns);
void getNumberOfNodesFromNodeFile(int *numberOfNodes);
double setAverageSeperationToOne(float4 *node, int numberOfNodes);
double setMinimumSeperationToOne(float4 *node, int numberOfNodes);
float4 setGeometricCenterToZero(float4 *node, int n);
void placeNodesFromAFile(float4 *node, int *numberOfNodes);
void printEdgeCosts(float4 *node, int n, double nodeAdjustmentFactor);
int factorial(int n);
void printPathOrder(int* path, int n);
double getPathCost(int *path, float4 *node, int type, int n);
void swap(int *path, int i, int j);
void heappermute(int* path, int m, float4 *node, int* exhaustivePath, double* minCost, int n);
double exhaustiveTSP(float4 *node, int* exhaustivePath, int n);
double nearestNeighborTSP(float4 *node, int* path, int n);
void setNbodyInitialConditions(float4 *node, float4 *pos, float4 *vel, float* mass, int n);
void drawNodes(float4* nodes, int n, float normalizingFactor);
void drawNbodyExtrusion(float4 *pos, float innerRadius, float outerRadius, int innerWallDirection, int outerWallDirection, int n);
void getPathNbody(float4 *pos, int* path, int n);
double findMinimumDistance(float4 *pos, int n);
double findPressureOnOuterWall(float4 *pos, float outerRadius, int n);
double NbodyExtrusionTSP(float4 *node, float4 *pos, float4 *vel, float4 *acc, float* mass, int* path, int n);
void drawFinalPicture(float4 *node, int *pathA, int *pathB, int n);
void getInputFromUser(int* scope, int* numberOfNodes, int* numberOfRuns, int* maxNumberOfRows, int* maxNumberOfColumns, unsigned int* srandSeed);
void control();

float x_machine_to_x_screen(int x)
{
	return( (2.0*x)/X_WINDOW-1.0 );
}

float y_machine_to_y_screen(int y)
{
	return( -(2.0*y)/Y_WINDOW+1.0 );
}

/*	Takes machine x and y which start in the upper left corner and go from zero to X_WINDOW
	left to right and form zero to Y_WINDOW top to bottom and transslates this into world
	points which are a X_MIN to X_MAX, Y_MIN to Y_MAX window.
*/
float x_machine_to_x_world(int x)
{
	float range;
	range = X_MAX - X_MIN;
	return( (range/X_WINDOW)*x + X_MIN );
}

float y_machine_to_y_world(int y)
{
	float range;
	range = Y_MAX - Y_MIN;
	return(-((range/Y_WINDOW)*y - Y_MAX));
}

/*	Take world  points to screen points
*/
float x_world_to_x_screen(float x)
{
	float range;
	range = X_MAX - X_MIN;
	return( -1.0 + 2.0*(x - X_MIN)/range );
}

float y_world_to_y_screen(float y)
{
	float range;
	range = Y_MAX - Y_MIN;
	return( -1.0 + 2.0*(y - Y_MIN)/range );
}

void openRawDataFile(int scope, int numberOfRuns)
{
	char tagName[50];
	char fileName[256];
	
	strcpy(tagName,"");
	if(scope == 2 || scope == 6)
	{
		strcat(tagName,"Random_TSP_Raw_Data_CSV");
	}
	if(scope == 4)
	{
		strcat(tagName,"Grid_TSP_Raw_Data_CSV");
	}
	
	snprintf(fileName, 256, "%s", tagName);
	
	RawDataFile = fopen(fileName, "wb");
  	
	fprintf(RawDataFile, "  Number of runs %d\n",numberOfRuns);
	
	if(scope == 2 || scope == 6)
	{
		fprintf(RawDataFile, "  Run  Nodes  Exact  NNeighbor  NBody\n\n");
	}
	if(scope == 4)
	{
		fprintf(RawDataFile, "  Run  Rows  Columns  Exact  NNeighbor  NBody\n\n");
	}
}

//This function adjustes the nodes so that the average seperation is 1.0
double setAverageSeperationToOne(float4 *node, int numberOfNodes)
{
	double nodeAdjustmentFactor;
	double sum;
	int numberOfEdges;
	double dx,dy;
	int i,j;
	
	sum = 0.0;
	for(i = 0; i < numberOfNodes; i++)
	{
		for(j = i + 1; j < numberOfNodes; j++)
		{
			dx = node[i].x-node[j].x;
			dy = node[i].y-node[j].y;
			sum += sqrt(dx*dx + dy*dy);
		}
	}
	
	numberOfEdges = ((numberOfNodes)*(numberOfNodes - 1))/2;
	nodeAdjustmentFactor = sum/numberOfEdges;
	
	for(int i = 0; i < numberOfNodes; i++)
	{
		node[i].x = node[i].x/nodeAdjustmentFactor;
		node[i].y = node[i].y/nodeAdjustmentFactor;
	}
	
	return(nodeAdjustmentFactor);
}

//This function adjustes the nodes so that the minimum seperation is 1.0
double setMinimumSeperationToOne(float4 *node, int numberOfNodes)
{
	double nodeAdjustmentFactor;
	double minimum;
	double dx,dy, d;
	int i,j;
	
	minimum = BIG_NUMBER;
	for(i = 0; i < numberOfNodes; i++)
	{
		for(j = i + 1; j < numberOfNodes; j++)
		{
			dx = node[i].x-node[j].x;
			dy = node[i].y-node[j].y;
			d = sqrt(dx*dx + dy*dy);
			if(d < minimum) minimum = d;
		}
	}
	
	nodeAdjustmentFactor = minimum;
	
	for(int i = 0; i < numberOfNodes; i++)
	{
		node[i].x = node[i].x/nodeAdjustmentFactor;
		node[i].y = node[i].y/nodeAdjustmentFactor;
	}
	
	return(nodeAdjustmentFactor);
}

float4 getBoundingBox(float4 *nodes, int n)
{
	float4 boundingBox = make_float4(nodes[0].x, nodes[0].y, nodes[0].x, nodes[0].y);

	for(int i = 1; i < n; i++)
	{
		if(nodes[i].x < boundingBox.x) boundingBox.x = nodes[i].x;
		if(nodes[i].y < boundingBox.y) boundingBox.y = nodes[i].y;
		if(nodes[i].x > boundingBox.z) boundingBox.z = nodes[i].x;
		if(nodes[i].y > boundingBox.w) boundingBox.w = nodes[i].y;
	}

	return(boundingBox);
}

float4 setGeometricCenterToZero(float4 *nodes, int n)
{
	float4 boundingBox = getBoundingBox(nodes, n);
	
	float4 geometricCenter;
	
	geometricCenter.x = 0.0;
	geometricCenter.y = 0.0;
	
	for(int i = 0; i < n; i++)
	{
		geometricCenter.x += nodes[i].x;
		geometricCenter.y += nodes[i].y;
	}
	
	geometricCenter.x /= (float)n;
	geometricCenter.y /= (float)n;
	
	
	for(int i = 0; i < n; i++)
	{
		nodes[i].x -= geometricCenter.x;
		nodes[i].y -= geometricCenter.y;
	}
	
	return(geometricCenter);
}

double findDistanceToOuterMostElement(float4 *element, int numberOfElements)
{
	double temp;
	double distanceToOutermostElement = 0.0;
	
	for(int i = 0; i < numberOfElements; i++)
	{
		temp = sqrt(element[i].x*element[i].x + element[i].y*element[i].y);
		if(temp > distanceToOutermostElement)
		{
			distanceToOutermostElement = temp;
		}
	}
	return(distanceToOutermostElement);
}

void printEdgeCosts(float4 *node, int n, float nodeAdjustmentFactor)
{
	double temp;
	for(int i = 0; i < n; i++)
	{
		for(int j = i + 1; j < n; j++)
		{	
			temp = sqrt((node[i].x-node[j].x)*(node[i].x-node[j].x) + (node[i].y-node[j].y)*(node[i].y-node[j].y))*nodeAdjustmentFactor;
			printf("edge cost [%d, %d] = %f\n", i, j, temp);
		}
	}
}

int factorial(int n)
{
	int outPut = n;
	
	for(int i = n-1; i > 0; i--)
	{
		outPut *= i;	
	}
	return(outPut);
}

void printPathOrder(int* path, int n)
{
	printf("  ");
	for(int i = 0; i < n-1; i++)
	{
		printf("%d->", path[i]);	
	}
	printf("%d", path[n-1]);
}

double getPathCost(int *path, float4 *node, int type, int n)
{
	double cost;
	int i, j, k;
	
	//Checking path validaty 
	for(i = 0; i < n; i++)
	{
		if(path[i] < 0 || (n-1) < path[i])
		{
			printf("\n\n  Error -> Path out of range! Type = %d", type);
			printf("\n  path[%d] = %d\n\n", i, path[i]);
			printf("\n\n  Good Bye.  \n\n");
			exit(0);
		}
		
		for(j = 0; j < i; j++)
		{
			if(path[i] == path[j])
			{
				printf("\n\n Error -> Path has a repeated index! Type = %d\n", type);
				printPathOrder(path, n);
				printf("\n\n");
				printf("\n\n  Good Bye.  \n\n");
				exit(0);
			}
		}
	}
	
	cost = 0.0;
	for(k = 0; k < n-1; k++)
	{
		i = path[k];
		j = path[k+1];
		cost += sqrt((node[i].x-node[j].x)*(node[i].x-node[j].x) + (node[i].y-node[j].y)*(node[i].y-node[j].y));
	}
	i = path[n-1];
	j = path[0];
	cost += sqrt((node[i].x-node[j].x)*(node[i].x-node[j].x) + (node[i].y-node[j].y)*(node[i].y-node[j].y));
	
	return(cost);
}

void swap(int *path, int i, int j)
{
	int temp;
	temp = path[i];
	path[i] = path[j];
	path[j] = temp;
}

void heappermute(int *path, int m, float4 *node, int *exhaustivePath, double *minCost, int n) 
{
	int i;
	double pathCost;
	int* pathPlus = (int*)malloc(n*sizeof(int));

	if (m == 1) 
	{
		pathPlus[0] = 0;
		for(i = 1; i < n; i++)
		{
			pathPlus[i] = path[i-1];	
		}
		
		pathCost = getPathCost(pathPlus, node, 1, n);
		
		if(PRINT_EXHAUSTIVE_PATHS == 1)
		{
			printf("\n");
			printPathOrder(pathPlus, n);
			printf(" cost = %f", pathCost);
		}
		
		if(pathCost < minCost[0])
		{
			minCost[0] = pathCost;
			for(i = 0; i < n; i++)
			{
				exhaustivePath[i] = pathPlus[i];	
			}
		}
    	}
	else 
	{
		for (i = 0; i < m; i++) 
		{
			heappermute(path, m-1, node, exhaustivePath, minCost, n);
			if (m % 2 == 1) 
			{
				swap(path, 0, m-1);
			}
			else 
			{
				swap(path, i, m-1);
			}
		}
	}
	free(pathPlus);
}

double exhaustiveTSP(float4 *node, int* exhaustivePath, int n)
{
	double cost[1];
	int* path = (int*)malloc((n-1)*sizeof(int));
	
	exhaustivePath[0] = 0;
	for(int i = 1; i < n; i++)
	{
		exhaustivePath[i] = i;
		path[i-1] = i;	
	}
	cost[0] = getPathCost(exhaustivePath, node, 1, n);
	
	heappermute(path, n-1, node, exhaustivePath, cost, n);
	free(path);
	return(cost[0]);
}

double nearestNeighborTSP(float4 *node, int* path, int n)
{
	int i, j, k, nextNode, nodeFound;
	double minCost, pathCost, edgeCost, maxEdgeCost;
	int* used = (int*)malloc(n*sizeof(int));
	
	maxEdgeCost = 0.0;
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			edgeCost = sqrt((node[i].x-node[j].x)*(node[i].x-node[j].x) + (node[i].y-node[j].y)*(node[i].y-node[j].y));
			if(edgeCost > maxEdgeCost) 
			{
				maxEdgeCost = edgeCost;
			}	
		}	
	}
	maxEdgeCost += 1.0;
	
	for(i = 0; i < n; i++)
	{
		used[i] = -1;	
	}
	
	path[0] = 0;
	used[0] = 1;
	
	k = 0;
	
	minCost = maxEdgeCost;
	while(k < n-1)
	{
		nodeFound = 0;
		for(j = 0; j < n; j++)
		{
			i = path[k];
			edgeCost = sqrt((node[i].x-node[j].x)*(node[i].x-node[j].x) + (node[i].y-node[j].y)*(node[i].y-node[j].y));
			if(edgeCost <= minCost && used[j] == -1)
			{
				minCost = edgeCost;
				nextNode = j;
				nodeFound = 1;
			}	
		}
		if(nodeFound == 0)
		{
			printf("\n\n  There was a problem in the nearest neighbor function. No next node was found.\n\n");
			printf("\n\n  Good Bye.  \n\n");
			exit(0);
		}
		nodeFound = 0;
		
		k++;
		path[k] = nextNode;
		used[nextNode] = 1;
		minCost = maxEdgeCost;
	}
	
	pathCost = getPathCost(path, node, 2, n);
	free(used);
	return(pathCost);
}

void setNbodyInitialConditions(float4 *node, float4 *pos, float4 *vel, float* mass, int n)
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

void moveAnyNodeOffDeadCenter(float4 *pos, int n)
{
	int i;

	for(i = 0; i < n; i++)
	{
		if( sqrt(pos[i].x*pos[i].x + pos[i].y*pos[i].y) < 0.001) 
		{
			pos[i].x = 0.001;
			pos[i].y = 0.001;
		}
	}
}

double findMinimumDistance(float4 *pos, int n)
{
	int i;
	double min;
	double temp = BIG_NUMBER;
	
	for(i = 0; i < n; i++)
	{
		temp = sqrt(pos[i].x*pos[i].x + pos[i].y*pos[i].y);
		if( temp < min) 
		{
			min = temp;
		}
	}
	return(min);
}

double findPressureOnOuterWall(float4 *pos, float outerRadius, int n)
{
	int i;
	double sum, temp;

	sum = 0.0;
	for(i = 0; i < n; i++)
	{
		temp = sqrt(pos[i].x*pos[i].x + pos[i].y*pos[i].y) - outerRadius;
		if( 0 < temp) 
		{
			sum += temp;
		}
	}
	return(sum*WALL_STRENGTH/(2.0*PI*outerRadius));
}



void drawNodes(float4 *nodes, int n, float normalizingFactor)
{
	int i;	
	glClear(GL_COLOR_BUFFER_BIT);
	
	glPointSize(5.0);
	glColor3f(1.0,0.0,0.0);
	for(i = 0; i < n; i++)
	{
		glBegin(GL_POINTS);
		glVertex2f(x_world_to_x_screen(nodes[i].x/normalizingFactor),y_world_to_y_screen(nodes[i].y/normalizingFactor));
		glEnd();

	}
	glFlush();
}

void drawNbodyExtrusion(float4 *pos, float innerRadius, float outerRadius, int innerWallDirection, int outerWallDirection, int n)
{
	int i;
	int lineAmount = 100;
	float normalizingFactor = outerRadius;

	outerRadius /= normalizingFactor;
	innerRadius /= normalizingFactor;

	glClear(GL_COLOR_BUFFER_BIT);
	
	GLfloat twicePi = 2.0f * PI;
	
	glLineWidth(1.0);
	if(innerWallDirection == -1) glColor3f(1.0,0.0,0.0);
	else if(innerWallDirection == 0) glColor3f(1.0,1.0,0.0);
	else glColor3f(0.0,0.0,1.0);
	glBegin(GL_LINE_LOOP);
		for(i = 0; i <= lineAmount;i++) 
		{ 
			glVertex2f(x_world_to_x_screen(innerRadius*cos(i*twicePi/lineAmount)), 
			           y_world_to_y_screen(innerRadius*sin(i*twicePi/lineAmount)));
		}
	glEnd();
	
	glLineWidth(1.0);
	if(outerWallDirection == -1) glColor3f(1.0,0.0,0.0);
	else if(outerWallDirection == 0) glColor3f(1.0,1.0,0.0);
	else glColor3f(0.0,0.0,1.0);
	glBegin(GL_LINE_LOOP);
		for(i = 0; i <= lineAmount;i++) 
		{ 
			glVertex2f(x_world_to_x_screen(outerRadius*cos(i*twicePi/lineAmount)), 
			           y_world_to_y_screen(outerRadius*sin(i*twicePi/lineAmount)));
		}
	glEnd();
	
	glPointSize(5.0);
	glColor3f(1.0,0.0,0.0);
	for(i = 0; i < n; i++)
	{
		glBegin(GL_POINTS);
		glVertex2f(x_world_to_x_screen(pos[i].x/normalizingFactor),y_world_to_y_screen(pos[i].y/normalizingFactor));
		glEnd();

	}
	
	glFlush();
}

__global__ void accelerationsNbody(float4 *node, float4 *pos, float4 *vel, float4 *acc, float *mass, float innerRadius, float outerRadius, int n)
{
	int j,ii;
    float3 forceSum;
    float4 nodeMe, posMe;
    float dx, dy, d, edgeLength; 
    float radius, forceMag;
    __shared__ float4 shnode[BLOCK], shPos[BLOCK];
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    forceSum.x = 0.0;
	forceSum.y = 0.0;
	
	nodeMe.x = node[id].x;
	nodeMe.y = node[id].y;
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
		    
    for(j=0; j < gridDim.x; j++)
    {
    	if(threadIdx.x + blockDim.x*j < n)
    	{
    		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
    		shnode[threadIdx.x] = node[threadIdx.x + blockDim.x*j];
    	}
    	__syncthreads();
   
		#pragma unroll 32
        for(int i = 0; i < blockDim.x; i++)	
        {
        	ii = i + blockDim.x*j;
		    if(ii != id && ii < n) 
		    {
				dx = shPos[i].x - posMe.x;
				dy = shPos[i].y - posMe.y;
				d = sqrtf(dx*dx + dy*dy);
				
				edgeLength = sqrtf((shnode[i].x - nodeMe.x)*(shnode[i].x - nodeMe.x) + (shnode[i].y - nodeMe.y)*(shnode[i].y - nodeMe.y));
				
				if(d <= edgeLength)
				{
					forceMag = -(edgeLength - d)*SLOPE_REPULSION;
	
				}
				else if(edgeLength < d && d < FORCE_CUTOFF)
				{
					forceMag =  MAG_ATRACTION/edgeLength;
				}
				else
				{
					forceMag = 0.0;
				}
				
				forceSum.x += forceMag*dx/d;
				forceSum.y += forceMag*dy/d;
		    }
		}
	}
	
	if(id < n)
	{
		// Forces between node and the walls
		dx = posMe.x;
		dy = posMe.y; 
		radius = sqrtf(dx*dx + dy*dy);
	
		if(radius < innerRadius) // Inside inner wall
		{
			forceMag = WALL_STRENGTH*(innerRadius - radius);
			forceSum.x += forceMag*dx/radius;
			forceSum.y += forceMag*dy/radius;
		}
		else if(radius > outerRadius) // Outside outer wall
		{
			forceMag = WALL_STRENGTH*(outerRadius - radius);
			forceSum.x += forceMag*dx/radius;
			forceSum.y += forceMag*dy/radius;
		}
		
		// Adding on damping force.
		forceSum.x += -DAMP*vel[id].x;
		forceSum.y += -DAMP*vel[id].y;
		
		// Creating the accelerations.
	    acc[id].x = forceSum.x/mass[id];
	    acc[id].y = forceSum.y/mass[id];
    }
}

__global__ void moveNbody(float4 *pos, float4 *vel, float4 *acc, float dt, int n)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < n)
    {
	    vel[id].x += acc[id].x*dt;
		vel[id].y += acc[id].y*dt;
		
		pos[id].x  += vel[id].x*dt;
		pos[id].y  += vel[id].y*dt;
    }
}

void getPathNbody(float4 *pos, int* path, int n)
{
	int i;
	double minValue;
	double *angle = (double*)malloc(n*sizeof(double));
	int *used = (int*)malloc(n*sizeof(int));
	
	for(i = 0; i < n; i++)
	{
		if(pos[i].x == 0 && pos[i].y == 0)
		{
			angle[i] = 0.0;
		}
		else if(pos[i].x >= 0 && pos[i].y >= 0)
		{
			if(pos[i].x == 0) angle[i] = 90.0;
			else angle[i] = atan(pos[i].y/pos[i].x)*180.0/PI;
		}
		else if(pos[i].x < 0 && pos[i].y >= 0)
		{
			angle[i] = 180.0 - atan(pos[i].y/(-pos[i].x))*180.0/PI;
		}
		else if(pos[i].x <= 0 && pos[i].y < 0)
		{
			if(pos[i].x == 0) angle[i] = 270.0;
			else angle[i] = 180.0 + atan(pos[i].y/pos[i].x)*180.0/PI;
		}
		else
		{
			angle[i] = 360.0 - atan(-pos[i].y/pos[i].x)*180.0/PI;
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

double NbodyExtrusionTSP(float4 *node, float4 *pos, float4 *vel, float4 *acc, float* mass, int* path, int n)
{
	int draw_count;
	int innerWallDirection, outerWallDirection;
	double pressure;
	double dr;
	float dt = TIME_STEP_SIZE;
	double pathCost;
	double time;
	float innerRadius, outerRadius;
	double stopSeperation;
	
	dim3 block, grid;
	float4 *posGPU, *velGPU, *accGPU; 
	float *massGPU;
	float4 *nodeGPU;
	
	// Setting up GPU parrellel structure.
	block.x = BLOCK;
	block.y = 1;
	block.z = 1;
	
	grid.x = (n-1)/block.x + 1;
	grid.y = 1;
	grid.z = 1;
	
	// Allocating memory.
	cudaMalloc( (void**)&nodeGPU, n *sizeof(float4));
	cudaMalloc( (void**)&posGPU, n *sizeof(float4));
	cudaMalloc( (void**)&velGPU, n *sizeof(float4));
	cudaMalloc( (void**)&accGPU, n *sizeof(float4));
	cudaMalloc( (void**)&massGPU, n *sizeof(float));
	
	// This is used to pause the program so you can setup to take a video of a run.
	if(DELAY_TO_RECORD == 1)
	{
		printf("\n\n  Enter a character to start\n\n"); getchar();
	}
	
	// Copying information up to the GPU.
	cudaMemcpy( nodeGPU, node, n *sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( posGPU, pos, n *sizeof(float4), cudaMemcpyHostToDevice );
    cudaMemcpy( velGPU, vel, n *sizeof(float4), cudaMemcpyHostToDevice );
    cudaMemcpy( massGPU, mass, n *sizeof(float), cudaMemcpyHostToDevice );
    
    moveAnyNodeOffDeadCenter(pos, n);
    stopSeperation = findMinimumDistance(pos, n)/2.0;
    innerRadius = 0.0;
    outerRadius = findDistanceToOuterMostElement(pos, n);
    drawNbodyExtrusion(pos, innerRadius, outerRadius, 0, 0, n);
    
    outerWallDirection = -1;
    innerWallDirection = 0;
    dr = outerRadius/STARTING_POINT_FOR_NUMBER_OF_MOVES;
	draw_count = 0;
	pressure = 0.0;
	while(innerRadius + stopSeperation < outerRadius)
	{
		outerRadius += dr*outerWallDirection;
		innerRadius += dr*innerWallDirection;
		time = 0.0;
		while(time < TIME_BETWEEN_WALL_MOVES)
		{		
			accelerationsNbody<<<grid, block>>>(nodeGPU, posGPU, velGPU, accGPU, massGPU, innerRadius, outerRadius, n);
			moveNbody<<<grid, block>>>(posGPU, velGPU, accGPU, dt, n);
			
			if(draw_count == STEPS_BETWEEN_VIEWING)
			{
				cudaMemcpy( pos, posGPU, n *sizeof(float4), cudaMemcpyDeviceToHost );
				drawNbodyExtrusion(pos, innerRadius, outerRadius, innerWallDirection, outerWallDirection, n);
				draw_count = 0;
			}
			draw_count++;
			time += dt;
		}
		
		cudaMemcpy( pos, posGPU, n *sizeof(float4), cudaMemcpyDeviceToHost );
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
		}
		else
		{
			innerWallDirection = 0;
			outerWallDirection = 1;
		}
	}
	
	getPathNbody(pos, path, n);
	pathCost = getPathCost(path, node, 3, n);
	
	return(pathCost);
}

void drawFinalPicture(float4 *nodes, int *pathA, int *pathB, int n)
{	
	int i;
	float outerRadius = findDistanceToOuterMostElement(nodes, n);
	float normalizingFactor = outerRadius; //((float)n)/IDEAL_NUMBER_OF_NODES;

	glClear(GL_COLOR_BUFFER_BIT);
	

	//Nearest Neighbor path
	if(DRAW_NEAREST_NEIGHBOR_PATH == 1)
	{
		glLineWidth(6.0);
		glColor3f(0.0,1.0,0.0);
		glBegin(GL_LINE_LOOP);
			for(i = 0; i < n; i++)
			{
				glVertex2f(x_world_to_x_screen(nodes[pathA[i]].x/normalizingFactor),y_world_to_y_screen(nodes[pathA[i]].y/normalizingFactor));
			}
		glEnd();
	}
	
	//Nbody Extrusion path
	if(DRAW_NBODY_EXTRUSION_PATH == 1)
	{
		glLineWidth(3.0);
		glColor3f(1.0,0.0,0.0);
		glBegin(GL_LINE_LOOP);
			for(i = 0; i < n; i++)
			{
				glVertex2f(x_world_to_x_screen(nodes[pathB[i]].x/normalizingFactor),y_world_to_y_screen(nodes[pathB[i]].y/normalizingFactor));
			}
		glEnd();
	}
	
	//Placing nodes
	glPointSize(8.0);
	glColor3f(1.0,1.0,1.0);
	for(i = 0; i < n; i++)
	{
		glBegin(GL_POINTS);
			glVertex2f(x_world_to_x_screen(nodes[i].x/normalizingFactor),y_world_to_y_screen(nodes[i].y/normalizingFactor));
		glEnd();
	}
	
	//Nearest neighbor start node 
	if(DRAW_NEAREST_NEIGHBOR_PATH == 1)
	{
		glPointSize(10.0);
		glColor3f(0.0,0.0,1.0);
		glBegin(GL_POINTS);
			glVertex2f(x_world_to_x_screen(nodes[pathA[0]].x/normalizingFactor),y_world_to_y_screen(nodes[pathA[0]].y/normalizingFactor));
		glEnd();
	}
	
	//Nbody extrution start and stop nodes
	if(DRAW_NBODY_EXTRUSION_PATH == 1)
	{
		glPointSize(10.0);
		glColor3f(0.0,1.0,0.0);
		glBegin(GL_POINTS);
			glVertex2f(x_world_to_x_screen(nodes[pathB[0]].x/normalizingFactor),y_world_to_y_screen(nodes[pathB[0]].y/normalizingFactor));
		glEnd();
	
		glColor3f(1.0,0.0,0.0);
		glBegin(GL_POINTS);
			glVertex2f(x_world_to_x_screen(nodes[pathB[n-1]].x/normalizingFactor),y_world_to_y_screen(nodes[pathB[n-1]].y/normalizingFactor));
		glEnd();
	}
	
	glFlush();
}

void display()
{
	time_t t;
	int scope, numberOfNodes, numberOfRuns;
	unsigned int srandSeed;
	int done;
	float4 *nodes;
	double nodeAdjustmentFactor;
	float4 geometricCenter;
	double distanceToOutermostNode;
	int *nearestNeighborPath, *NbodyExtrusionPath;
	float4 *posNbody, *velNbody, *accNbody; 
	float *massNbody;
	double nearestNeighborCost, NbodyExtrusionCost;
	int nodeCheck;
	double temp;
	
	double totalNearestNeighborCost = 0.0;
	double totalNbodyExtrusionCost = 0.0;
	double totalPercentErrorNearestNeighbor = 0.0;
	double totalPercentErrorNbodyExtrusion = 0.0;
	double NbodyExtrusionVSNearestNeighbor = 0.0;
	
	numberOfRuns = 1;
	for(int i = 0; i < numberOfRuns; i++)
	{	
		printf("\n\n\n  ********************* Intermediate Run %d ********************* ", i+1);
		
		nearestNeighborCost = BIG_NUMBER;
		NbodyExtrusionCost = BIG_NUMBER;
		
		//Alocating memory
		int numberOfNodes = getNumberOfLines("./datasets/att48/coords.txt");
		float4 *nodes = (float4*)malloc((numberOfNodes)*sizeof(float4));;
		loadNodesFromFile("./datasets/att48/coords.txt", nodes);

		nearestNeighborPath = (int*)malloc((numberOfNodes)*sizeof(int));
		NbodyExtrusionPath = (int*)malloc((numberOfNodes)*sizeof(int));

		posNbody = (float4*)malloc((numberOfNodes)*sizeof(float4));
		velNbody = (float4*)malloc((numberOfNodes)*sizeof(float4));
		accNbody = (float4*)malloc((numberOfNodes)*sizeof(float4));
		massNbody = (float*)malloc((numberOfNodes)*sizeof(float4));

		//Adjusting nodes
		geometricCenter = setGeometricCenterToZero(nodes, numberOfNodes);
		printf("\n\n  The geometric center of the nodes = (%f, %f)", geometricCenter.x, geometricCenter.y);
		
		distanceToOutermostNode = findDistanceToOuterMostElement(nodes, numberOfNodes);
		printf("\n  The distance to the outermost node from the geometric center pre adjustment is %f", distanceToOutermostNode);
		
		//nodeAdjustmentFactor = setAverageSeperationToOne(node, numberOfNodes);
		nodeAdjustmentFactor = setMinimumSeperationToOne(nodes, numberOfNodes);
		printf("\n  The node adjustment factor = %f", nodeAdjustmentFactor);
		
		distanceToOutermostNode = findDistanceToOuterMostElement(nodes, numberOfNodes);
		printf("\n  The distance to the outermost node from the geometric center post adjustment is %f", distanceToOutermostNode);
		
		//Drawing the adjusted nodes on the screen.
		drawNodes(nodes, numberOfNodes, distanceToOutermostNode); 
		//Printing the edge costs (lengths in this case)
		if(PRINT_EDGE_COST == 1)
		{
			printEdgeCosts(nodes, numberOfNodes, nodeAdjustmentFactor);
		}
				
		//Finding nearest neighbor cost
		printf("\n\n  Running the nearest nieghbor algorithm.");
		nearestNeighborCost = nearestNeighborTSP(nodes, nearestNeighborPath, numberOfNodes);
		printf("\n  The nearest nieghbor algorithm is done.");
		
		//Running n-body extrusion code
		printf("\n\n  Running the N-body extrusion algorithm."); 
		printf("  \n"); //I had to enter this carage return so it would print the line above before it started the algorithm
		setNbodyInitialConditions(nodes, posNbody, velNbody, massNbody, numberOfNodes);
		NbodyExtrusionCost = NbodyExtrusionTSP(nodes, posNbody, velNbody, accNbody, massNbody, NbodyExtrusionPath, numberOfNodes);
		printf("  The N-body extrusion algorithm is done.");
		
		//Unadjusting costs
		nearestNeighborCost *= nodeAdjustmentFactor;
		NbodyExtrusionCost *= nodeAdjustmentFactor;
		
		totalNearestNeighborCost += nearestNeighborCost;
		totalNbodyExtrusionCost += NbodyExtrusionCost;
		
/* 		//Sanity check
		if(nearestNeighborCost < exhaustiveCost - SMALL_NUMBER)
		{
			printf("\n\n  Nearest Neighbor cost (%f) is smaller than exhaustive cost (%f). Something is wrong!\n",nearestNeighborCost, exhaustiveCost);
			printf("\n\n  Good Bye.  \n\n");
			exit(0);
		}
		if(NbodyExtrusionCost < exhaustiveCost - SMALL_NUMBER)
		{
			printf("\n\n  Nbody Extrution cost (%f) is smaller than exhaustive cost (%f). Something is wrong!\n",NbodyExtrusionCost, exhaustiveCost);
			printf("\n\n  Good Bye.  \n\n");
			exit(0);
		}
		*/
		printf("\n\n  --------------------- Intermediate Run Results --------------------- ");
		
		// This is for debugging
		if(PRINT_PATHS == 1)
		{	
			printf("\n\n  The nearest neighbor path is: "); 
			printPathOrder(nearestNeighborPath, numberOfNodes); 
			printf(" cost = %f", nearestNeighborCost);
			
			printf("\n\n  The Nbody extrusion path is : "); 
			printPathOrder(NbodyExtrusionPath, numberOfNodes); 
			printf(" cost = %f", NbodyExtrusionCost);
		}
		
		// Printing out the single run stats and acumulating the multiple run info to create final stats.
		// Stephen your stat collection should go here.
		{			
			NbodyExtrusionVSNearestNeighbor += (nearestNeighborCost - NbodyExtrusionCost)/nearestNeighborCost;
		}
		

		drawFinalPicture(nodes, nearestNeighborPath, NbodyExtrusionPath, numberOfNodes);
		free(nodes);
		free(nearestNeighborPath);
		free(NbodyExtrusionPath);
		free(posNbody);
		free(velNbody);
		free(accNbody);
		free(massNbody);
	}
	
	printf("\n\n\n  $$$$$$$$$$$$$$$$$$$$$$$$$ Final results $$$$$$$$$$$$$$$$$$$$$$$$$$$$$");
	
	// Printing out the final acumulated stats.
	// Stephen your stat final stats should go here.
/* 	if(exhaustiveCost < 0.0)
	{
		printf("\n\n  The average value of the nearest neighbor method was %f on %d run(s).", totalNearestNeighborCost/numberOfRuns, numberOfRuns);
		printf("\n\n  The average value of the Nbody extrution method was %f on %d run(s).", totalNbodyExtrusionCost/numberOfRuns, numberOfRuns);
	}
	else
	{
		printf("\n\n  The average percent error of the nearest neighbor method was %f on %d runs.", totalPercentErrorNearestNeighbor/(float)numberOfRuns, numberOfRuns);
		printf("\n  The average percent error of the Nbody extrution method was %f on %d runs", totalPercentErrorNbodyExtrusion/(float)numberOfRuns, numberOfRuns);
	}
	 */
	NbodyExtrusionVSNearestNeighbor = 100.0*NbodyExtrusionVSNearestNeighbor/(float)numberOfRuns;
	if(NbodyExtrusionVSNearestNeighbor >= 0)
	{
		printf("\n\n  The Nbody ectrusion method was on average %f percent better than the nearest neighbor method on %d run(s).", NbodyExtrusionVSNearestNeighbor, numberOfRuns);
	}
	else
	{
		printf("\n\n  The Nbody ectrusion method was on average %f percent worse than the nearest neighbor method on %d run(s).", -NbodyExtrusionVSNearestNeighbor, numberOfRuns);
	}
	
	printf("\n\nDone\n");
}

void mouse(int button, int state, int mx, int my)
{
	clicked = 1;
	glutPostRedisplay();
}


int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitWindowSize(X_WINDOW,Y_WINDOW);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Traveling Salesman Problem");
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMainLoop();
}



    

