// OpenGL Graphics includes
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "./headers/dataio.h"
#include "./headers/rendering.h"
#include "./headers/drawing.h"
#include "./headers/density.cuh"

#define DIM 1024
#define FP "./datasets/att48/coords.txt"

// Move these to be initialized in main()
int numberOfNodes = 0;
float2 *nodes = nullptr;

float4 boundingBox;
float2 geometricCenter;

int mouse_x, mouse_y;
GLFWwindow* window;

void delay(int number_of_seconds) 
{ 
    // Converting time into milli_seconds 
    int milli_seconds = 1000 * number_of_seconds; 
  
    // Storing start time 
    clock_t start_time = clock(); 
  
    // looping till required time is not achieved 
    while (clock() < start_time + milli_seconds)
        ; 
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

float rnd(float x)
{
    return(x*rand() / RAND_MAX);
}

void render()
{
    glClear(GL_COLOR_BUFFER_BIT);
    int b = 8;
    float scale = 1.0;
    float colorNodes[3] = {0.1, 0.1, 0.1};
    float colorCenter[3] = {0.0, 1.0, 0.2};
    //drawDensity(nodes, numberOfNodes, b, scale);
    drawPoints(nodes, numberOfNodes, 3.0, colorNodes);
    //drawGrid(2*scale/b, 2*scale/b, scale);
    //drawPoint(geometricCenter, 2.5, colorCenter);
    
    glfwSwapBuffers(window);
    delay(50);
    
    for(int i = 0; i < numberOfNodes; i++)
    {
        nodes[i].x += rnd(0.06) - 0.03;
        nodes[i].y += rnd(0.06) - 0.03;
    }
}

// Mouse callback function
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    mouse_x = (int)xpos;
    mouse_y = (int)ypos;
}

// Error callback for GLFW
void error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int argc, char** argv)
{
    printf("Starting TSP Visualization...\n");
    
    // Initialize data INSIDE main() with proper error checking
    printf("Checking for data file: %s\n", FP);
    FILE* test_file = fopen(FP, "r");
    if (!test_file) {
        printf("WARNING: Cannot find data file: %s\n", FP);
        printf("Creating test data instead...\n");
        
        // Create test data
        numberOfNodes = 6;
        nodes = (float2*)malloc(numberOfNodes * sizeof(float2));
        if (!nodes) {
            printf("ERROR: Failed to allocate memory\n");
            return -1;
        }
        
        // Create a simple hexagon pattern
        nodes[0] = make_float2(0.0f, 0.0f);
        nodes[1] = make_float2(1.0f, 0.0f);  
        nodes[2] = make_float2(0.5f, 0.866f);
        nodes[3] = make_float2(-0.5f, 0.866f);
        nodes[4] = make_float2(-1.0f, 0.0f);
        nodes[5] = make_float2(-0.5f, -0.866f);
        
        printf("✓ Created %d test nodes\n", numberOfNodes);
    } else {
        fclose(test_file);
        printf("✓ Found data file\n");
        
        // Get number of lines and allocate memory
        numberOfNodes = getNumberOfLines(FP);
        if (numberOfNodes <= 0) {
            printf("ERROR: Invalid number of nodes: %d\n", numberOfNodes);
            return -1;
        }
        printf("✓ Number of nodes: %d\n", numberOfNodes);
        
        nodes = (float2*)malloc(numberOfNodes * sizeof(float2));
        if (!nodes) {
            printf("ERROR: Failed to allocate memory for %d nodes\n", numberOfNodes);
            return -1;
        }
        
        // Load nodes from file
        loadNodesFromFile(FP, nodes);    
        printf("✓ Loaded node positions from file\n");
    }

    // Set error callback
    glfwSetErrorCallback(error_callback);
    
    printf("Initializing GLFW...\n");
    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "ERROR: Failed to initialize GLFW\n");
        free(nodes);
        return -1;
    }
    printf("✓ GLFW initialized\n");
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    printf("Creating window...\n");
    // Create window
    window = glfwCreateWindow(DIM, DIM, "Traveling Salesman Problem", NULL, NULL);
    if (!window)
    {
        fprintf(stderr, "ERROR: Failed to create GLFW window\n");
        glfwTerminate();
        free(nodes);
        return -1;
    }
    printf("✓ Window created\n");
    
    // Set window position
    glfwSetWindowPos(window, 0, 0);
    
    // Make the OpenGL context current
    glfwMakeContextCurrent(window);
    
    printf("Initializing GLEW...\n");
    // Initialize GLEW
    GLenum glewError = glewInit();
    if (glewError != GLEW_OK)
    {
        fprintf(stderr, "ERROR: Failed to initialize GLEW: %s\n", glewGetErrorString(glewError));
        glfwTerminate();
        free(nodes);
        return -1;
    }
    printf("✓ GLEW initialized\n");
    
    // Set callbacks
    glfwSetCursorPosCallback(window, mouse_callback);
    
    // Enable V-Sync
    glfwSwapInterval(1);
    
    printf("Processing node data...\n");
    boundingBox = getBoundingBox(nodes, numberOfNodes);
    printf("✓ Got bounding box\n");

    float dx = boundingBox.z - boundingBox.x;
    float dy = boundingBox.w - boundingBox.y;

    linearTransformPoints(nodes, nodes, numberOfNodes, -0.5, 0.75, dx, dy);
    printf("✓ Transformed node positions\n");

    geometricCenter = getGeometricCenter(nodes, numberOfNodes);
    printf("✓ Got geometric center\n");
    
    // Set OpenGL state
    printf("Setting up OpenGL state...\n");
    glClearColor(1.0, 1.0, 1.0, 0.1);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    printf("✓ OpenGL configured\n");
    
    printf("Starting main loop...\n");
    printf("Press ESC to exit\n");
    
    // Main render loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll for and process events
        glfwPollEvents();
        
        // Render
        render();
        
        // Check for escape key to exit
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
    
    printf("Cleaning up...\n");
    // Cleanup
    free(nodes);
    glfwTerminate();
    printf("Program ended normally\n");
    return 0;
}