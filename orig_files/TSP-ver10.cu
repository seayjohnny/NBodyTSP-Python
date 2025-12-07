// Fix Windows header conflicts
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // Add for memset

#include "./headers/vecmath.cuh"
#include "./headers/dataio.cuh"
#include "./headers/lintrans.cuh"
#include "./headers/structs.cuh"
#include "./headers/arrays.h"
#include "./headers/nbodyrender.cuh"
#include "./headers/runtimer.cuh"

//TODO : nbodyrender.h
//TODO : arrmath.h

#define BLOCK 256
#define FORCE_CUTOFF 100000.0

#define DRAW 1

__global__ void initBubbles(RunState* rs)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int id = x + y*gridDim.x*blockDim.x;

    if(rs->densities[id] > 3)
    {
        rs->bubbles[id].center = rs->densityCenters[id];
        rs->bubbles[id].strength = 20000.0;
    }
}

__global__ void getDensity(RunState* rs, float dr)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int id = x + y*gridDim.x*blockDim.x;

    float xr[2] = {rs->range[blockIdx.x], rs->range[blockIdx.x+1]};
    float yr[2] = {rs->range[B-1-blockIdx.y], rs->range[B-1-blockIdx.y+1]};
    float xBar = 0.0;
    float yBar = 0.0;
    if(id == 0){
        for(int i = 0; i < B*B; i++)
        {
            rs->densities[i] = 0;
        }
    }
    __syncthreads();

    for(int i = 0; i < rs->numberOfNodes; i++)
    {
        if(rs->nodes[i].pos.x >= xr[0] && rs->nodes[i].pos.x <= xr[1])
        {
            if(rs->nodes[i].pos.y >= yr[0] && rs->nodes[i].pos.y <= yr[1])
            {
                rs->densities[id] += 1;
                xBar += rs->nodes[i].pos.x;
                yBar += rs->nodes[i].pos.y;
            }
        }
    }
    if(rs->densities[id])
    {
        rs->densityCenters[id].x = xBar/rs->densities[id];
        rs->densityCenters[id].y = yBar/rs->densities[id];
    }

    if(rs->bubbles[id].strength)
    {
        float d = mag(rs->bubbles[id].center) + rs->bubbles[id].radius;
        if(d < rs->outerWall.radius) {
            rs->bubbles[id].radius += dr*rs->innerWall.direction;
        }
    }
}

__host__ __device__ float getPressure(RunState* rs)
{
    int i;
    float sum, temp;
    
    sum = 0.0;
    for(i = 0; i<rs->numberOfNodes;i++)
    {
        temp = mag(rs->nodes[i].pos) - rs->outerWall.radius;
        if(0 < temp)
        {
            sum += temp;
        }
    }
    return(sum*(rs->outerWall.strength)/(2.0*PI*rs->outerWall.radius)); //PI is defined in vecmath.cuh
}

__global__ void adjustForPressure(RunState* rs)
{
    rs->pressure = getPressure(rs);
    if(rs->pressure < rs->lowerPressureLimit)
    {
        rs->innerWall.direction = 0;
        rs->outerWall.direction = -1;
    }
    else if(rs->pressure < rs->upperPressureLimit)
    {
        rs->innerWall.direction = 1;
        rs->outerWall.direction = 0;
        if(rs->bubbleFlag == -1) rs->bubbleFlag = 1;
    } 
    else
    {
        rs->innerWall.direction = 0;
        rs->outerWall.direction = 1;
    }
}

__global__ void nBodyStep(RunState* rs, float dt, float dr)
{
    __shared__ float2 shPos[N], shInitPos[N];
    float d, edgeLength;

    int id = threadIdx.x;
    
    if(id < N)
    {
        float2 pos = shPos[id] = rs->nodes[id].pos;
        float2 initPos = shInitPos[id] = rs->nodes[id].initpos;

        float2 force = {0, 0};
        double forceMag;
        float h, c, radius;

        __syncthreads();
        for(int i = 0; i < N; i++)
        {
            if(i != id)
            {
                d = dist(shPos[i], pos);
                edgeLength = dist(shInitPos[i], initPos);

                h = rs->m*(powf(powf(rs->q/rs->p, 1/(rs->q-rs->p))*edgeLength, rs->p))/(1 - rs->p/rs->q);
                c = powf(edgeLength/d, rs->q-rs->p);
                forceMag = (c-1)*h/powf(d, rs->p);
                
                force += make_float2(forceMag*(shPos[i].x - pos.x)/d, forceMag*(shPos[i].y - pos.y)/d);
            }
        }
        __syncthreads();
        radius = mag(pos);
        if(radius < rs->innerWall.radius)
        {
            forceMag = rs->innerWall.strength*(rs->innerWall.radius - radius);
            force += make_float2(forceMag*pos.x/radius, forceMag*pos.y/radius);
        }
        else if(radius > rs->outerWall.radius)
        {
            forceMag = rs->outerWall.strength*(rs->outerWall.radius - radius);
            force += make_float2(forceMag*pos.x/radius, forceMag*pos.y/radius);
        }

        float dx, dy;
        for(int i = 0; i < B*B;i++)
        {
            if(rs->bubbles[i].strength)
            {
                dx = pos.x - rs->bubbles[i].center.x;
                dy = pos.y - rs->bubbles[i].center.y;
                if(abs(dx) < rs->bubbles[i].radius && abs(dy) < rs->bubbles[i].radius)
                {
                    radius = sqrt(dx*dx + dy*dy);
                    if(radius < rs->bubbles[i].radius)
                    {
                        forceMag = rs->bubbles[i].strength*(rs->bubbles[i].radius - radius);
                        force += make_float2(forceMag*dx/radius, forceMag*dy/radius);
                    }
                }
            }
        }
        __syncthreads();
        force += make_float2(-rs->damp*rs->nodes[id].vel.x, -rs->damp*rs->nodes[id].vel.y);
        if(force.x > FORCE_CUTOFF) force.x = 0;
        if(force.y > FORCE_CUTOFF) force.y = 0;
        rs->nodes[id].acc = force/rs->nodes[id].mass;
       
        __syncthreads();
        rs->nodes[id].vel += rs->nodes[id].acc*dt;
        rs->nodes[id].pos += rs->nodes[id].vel*dt;
    }

    if(id == 0)
    {
        rs->innerWall.radius += dr*rs->innerWall.direction;
        rs->outerWall.radius += dr*rs->outerWall.direction;
    }
}

float getPathCost(RunState rs, RunState os)
{
    float cost = 0.0;
    float dx, dy;
    int n = rs.numberOfNodes;

    for(int i = 0; i < n; i++)
    {
        dx = os.nodes[rs.nBodyPath[i%n]].initpos.x - os.nodes[rs.nBodyPath[(i+1)%n]].initpos.x;
        dy = os.nodes[rs.nBodyPath[i%n]].initpos.y - os.nodes[rs.nBodyPath[(i+1)%n]].initpos.y;
        cost += sqrt(dx*dx + dy*dy);
    }

    return cost;
}

void normalizeNodes(RunState *rs)
{
    for(int i = 0; i < N ;i++)
    {
        rs->nodes[i].pos /= rs->factor;
    }
}

void normalizeDensityCenters(RunState *rs)
{
    for(int i = 0; i < B*B;i++)
    {
        rs->densityCenters[i] /= rs->factor;
    }
}

void normalizeCircles(RunState* rs)
{
    rs->factor = rs->outerWall.radius;

    rs->innerWall.radius /= rs->factor;
    rs->outerWall.radius /= rs->factor;
    for(int i = 0; i < B*B; i++)
    {
        rs->bubbles[i].radius /= rs->factor;
    }
}

void debugRenderState(RunState& rs) {
    printf("\n--- DEBUG: Render State ---\n");
    printf("Number of nodes: %d\n", rs.numberOfNodes);
    printf("Factor: %f\n", rs.factor);
    printf("Inner wall radius: %f\n", rs.innerWall.radius);
    printf("Outer wall radius: %f\n", rs.outerWall.radius);
    
    printf("First 5 node positions:\n");
    for(int i = 0; i < 5 && i < rs.numberOfNodes; i++) {
        printf("  Node %d: pos(%.6f, %.6f) initpos(%.6f, %.6f)\n", 
               i, rs.nodes[i].pos.x, rs.nodes[i].pos.y,
               rs.nodes[i].initpos.x, rs.nodes[i].initpos.y);
    }
    
    // Check if any bubbles have strength
    int bubbleCount = 0;
    for(int i = 0; i < B*B; i++) {
        if(rs.bubbles[i].strength != 0) bubbleCount++;
    }
    printf("Active bubbles: %d\n", bubbleCount);
    printf("-------------------------\n\n");
}

// Add this simple test drawing function
void testDraw() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw a simple test triangle to verify OpenGL is working
    glColor3f(1.0, 0.0, 0.0); // Red
    glBegin(GL_TRIANGLES);
    glVertex2f(-0.5f, -0.5f);
    glVertex2f(0.5f, -0.5f);
    glVertex2f(0.0f, 0.5f);
    glEnd();
    
    // Draw some test points
    glColor3f(0.0, 1.0, 0.0); // Green
    glPointSize(10.0);
    glBegin(GL_POINTS);
    glVertex2f(-0.3f, -0.3f);
    glVertex2f(0.3f, -0.3f);
    glVertex2f(0.0f, 0.3f);
    glEnd();
    
    glfwSwapBuffers(window0);
}

int main(int argc, char** argv)
{
    printf("Starting TSP N-Body Simulation...\n");
    
    // Check for data files in order of preference
    const char* datafiles[] = {
        "./datasets/rand128/coords.txt",
        "./datasets/att48/coords.txt", 
        "./datasets/bay29/coords.txt"
    };
    const char* datafile = NULL;
    
    for(int i = 0; i < 3; i++) {
        FILE* test_file = fopen(datafiles[i], "r");
        if(test_file) {
            datafile = datafiles[i];
            fclose(test_file);
            printf("Found data file: %s\n", datafile);
            break;
        }
    }
    
    if(!datafile) {
        printf("ERROR: Cannot find any data files!\n");
        printf("Please ensure one of these files exists:\n");
        for(int i = 0; i < 3; i++) {
            printf("  - %s\n", datafiles[i]);
        }
        return -1;
    }
    
    int n = getNumberOfNodes(datafile);
    printf("Number of nodes: %d (Expected: %d)\n", n, N);
    if(n != N) {
        printf("ERROR: Node count mismatch!\n");
        return 1;
    }

    RunState o_rs;
    RunState h_rs;
    RunState* d_rs;
    cudaMalloc(&d_rs, sizeof(RunState));
    memset(&h_rs, 0, sizeof(h_rs));
    memset(&o_rs, 0, sizeof(o_rs));

    printf("Initializing parameters...\n");
    h_rs.numberOfNodes = n;
    h_rs.m = -0.05;
    h_rs.p = 9.073103;
    h_rs.q = 13.695556;
    h_rs.damp = 20.0;
    h_rs.mass = 80.0;
    h_rs.lowerPressureLimit = 1.0;
    h_rs.upperPressureLimit = 10.0;
    h_rs.innerWall = WallDefault;
    h_rs.innerWall.direction = 1;
    h_rs.innerWall.strength = 20000.0;
    h_rs.outerWall = WallDefault;
    h_rs.outerWall.radius = 1.0;
    h_rs.outerWall.direction = 0;
    h_rs.outerWall.strength = 20000.0;
    h_rs.progress = 0.0;
    h_rs.bubbleFlag = -1;
    memset(h_rs.nBodyPath, 0, n*sizeof(h_rs.nBodyPath[0]));
    linspace(h_rs.range, -1.0, 1.0, B+1, 1);

    for(int i = 0; i < B*B; i++)
    {
        h_rs.bubbles[i] = WallDefault;
    }
    
    printf("Loading node data from %s...\n", datafile);
    loadNodes(h_rs.nodes, datafile);
    loadNodes(o_rs.nodes, datafile);

    printf("Processing node positions...\n");
    shiftNodes(h_rs.nodes, n, getGeometricCenter(h_rs.nodes, n)*(-1.0));
    normalizeNodePositions(h_rs.nodes, n);
    resetNodeInitPositions(h_rs.nodes, n);

    printf("=== DEBUGGING DATA LOADING ===\n");
    debugRenderState(h_rs);

    dim3 grid;
    grid.x = B;
    grid.y = B;
    grid.z = 1;

    cudaMemcpy(d_rs, &h_rs, sizeof(RunState), cudaMemcpyHostToDevice);
    
    bool shouldContinue = true;
    
    // Right before the drawing loop, add a test:
    if(DRAW) {
        printf("Testing basic OpenGL drawing...\n");
        glfwMakeContextCurrent(window0);
        testDraw();
        printf("Test drawing completed. Press any key...\n");
        getchar(); // Wait for user input
    }

    if(DRAW)
    {
        printf("Initializing visualization...\n");
        nBodyRenderInit(argc, argv, h_rs);
        printf("Visualization initialized. Click in window to start simulation...\n");
        while(waitForMouseClick()) {
            if(!getWindowState()) {
                printf("Window closed, exiting...\n");
                shouldContinue = false;
                break;
            }
        }
    }

    if(shouldContinue) {
        printf("Starting n-body simulation...\n");
        float innerRadius = 0.0;
        float outerRadius = 1.0;
        float dt = 0.0001;
        float dr = outerRadius/20000;
        float t = 0.0;
        int drawCount = 0;
        double timer;
        startTimer(&timer);
        
        while(innerRadius < outerRadius-dr && shouldContinue)
        {
            t = 0.0;
            while(t < 1.0 && shouldContinue)
            {
                printf("\r Running n-body extrusion. Progress: %.1f%%", 
                       100.0*innerRadius/outerRadius);
                fflush(stdout);
                
                nBodyStep<<<1,n>>>(d_rs, dt, dr);
                adjustForPressure<<<1,1>>>(d_rs);
                t += dt;
                
                if(drawCount == 100 && DRAW)
                {
                    cudaMemcpy(&h_rs, d_rs, sizeof(RunState), cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    normalizeCircles(&h_rs);
                    normalizeNodes(&h_rs);
                    normalizeDensityCenters(&h_rs);
                    
                    if(getWindowState()) {
                        nBodyRenderUpdate(h_rs);
                    } else {
                        printf("\nWindow closed, stopping simulation...\n");
                        shouldContinue = false;
                        break;
                    }
                    drawCount = 0;
                }
                if(DRAW) drawCount++;
            }
            
            if(shouldContinue) {
                cudaMemcpy(&innerRadius, &d_rs->innerWall.radius, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&outerRadius, &d_rs->outerWall.radius, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_rs.bubbleFlag, &d_rs->bubbleFlag, sizeof(int), cudaMemcpyDeviceToHost);
            }
        }
        
        if(shouldContinue) {
            printf("\r N-body extrusion completed. ✓                    \n");
            cudaMemcpy(&h_rs, d_rs, sizeof(RunState), cudaMemcpyDeviceToHost);

            getNBodyPath(h_rs.nBodyPath, h_rs.nodes, n);
            h_rs.nbodyCost = getPathCost(h_rs, o_rs);
            h_rs.optimalCost = 33523.708;
            h_rs.percentDiff = 100*(h_rs.nbodyCost - h_rs.optimalCost)/h_rs.optimalCost;
            
            printf("N-body path cost: %.2f\n", h_rs.nbodyCost);
            printf("Optimal path cost: %.2f\n", h_rs.optimalCost);
            printf("Percent difference: %.2f%%\n", h_rs.percentDiff);
            
            endTimer(&timer);
            h_rs.runTime = timer;
            h_rs.drawn = DRAW;
            logRun(&h_rs, "./runlog2.txt");

            if(DRAW) {
                printf("Showing final path. Close window or press ESC to exit...\n");
                while(getWindowState()) {
                    nBodyDrawPath(h_rs);
                    // Add ESC key check if available
                    break; // Remove infinite loop for now
                }
            }
        }
    }
    
    // Cleanup
    cudaFree(d_rs);
    if(DRAW) {
        cleanupRendering();
    }
    printf("Program completed.\n");
    return 0;
}