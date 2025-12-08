#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <stdio.h>

#include "structs.cuh"
#include "lintrans.cuh"

#define DIM 768
#define TARGET_FPS 60
#define MAX_STRING 20

RunState rs;
float2 p[N];

double TimePerFrame = 1000/TARGET_FPS;
int CurrentFrame = 0;
int startTime;

GLFWwindow* window0; // Main visualization window
GLFWwindow* window1; // Info window
int closed = 0;
int clicked = 0;

float scale = 0.9;

float Aged[3] = {1.0, 253.0/255.0, 240.0/255.0};
float LightGray0[3] = {220.0/255.0, 220.0/255.0, 220.0/255.0};
float LightGray1[3] = {206.0/255.0, 206.0/255.0, 206.0/255.0};
float LightGray2[3] = {188.0/255.0, 188.0/255.0, 188.0/255.0};
float DarkGray2[3] = {50.0/255.0, 50.0/255.0, 50.0/255.0};
float DarkGray3[3] = {89.0/255.0, 89.0/255.0, 89.0/255.0};
float Green[3] = {66.0/255.0, 244.0/255.0, 143.0/255.0};
float Red[3] = {244.0/255.0, 65.0/255.0, 65.0/255.0};
float Blue[3] = {109.0/255.0, 158.0/255.0, 235.0/255.0};
float White[3] = {1.0, 1.0, 1.0};
float Black[3] = {0.0, 0.0, 0.0};

// Error callback for GLFW
void error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Window close callback
void window_close_callback(GLFWwindow* window)
{
    closed = 1;
}

// Mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        clicked = 1;
    }
}

void drawGrid(float dx, float dy, float gridLineColor[3], float xAxisColor[3], float yAxisColor[3])
{
    // Draw bounding box
    glLineWidth(0.5);
    (gridLineColor==NULL) ? glColor3f(1.0, 1.0, 1.0):glColor3f(gridLineColor[0], gridLineColor[1], gridLineColor[2]);
    glBegin(GL_LINES);
    for(float x = -scale; x <= scale+0.01; x+=dx)
    {
        glVertex2f(x, scale);
        glVertex2f(x, -scale);
    }

    for(float y = -scale; y <= scale+0.01; y+=dy)
    {
        glVertex2f(-scale, y);
        glVertex2f(scale, y);
    }

    (xAxisColor==NULL) ? glColor3f(1.0, 1.0, 1.0):glColor3f(xAxisColor[0], xAxisColor[1], xAxisColor[2]);
    glVertex2f(0.0, scale);
    glVertex2f(0.0, -scale);

    (yAxisColor==NULL) ? glColor3f(1.0, 1.0, 1.0):glColor3f(yAxisColor[0], yAxisColor[1], yAxisColor[2]);
    glColor3f(0.1, 0.4, 0.4);
    glVertex2f(scale, 0.0);
    glVertex2f(-scale, 0.0);

    glEnd();
}

void drawPoint(float2 point, float size, float color[3])
{
    glPointSize(size+10);
    (color==NULL) ? glColor3f(1.0, 1.0, 1.0):glColor3f(color[0], color[1], color[2]);
    glBegin(GL_POINTS);
    glVertex2f(point.x,point.y);
    glEnd();
}

void drawPoints(float2* points, int n, float pointSize, float color[3])
{
    glPointSize(pointSize);
    (color==NULL) ? glColor3f(1.0, 1.0, 1.0):glColor3f(color[0], color[1], color[2]);
    for(int i = 0; i < n; i++)
	{
        glBegin(GL_POINTS);
        glVertex2f(points[i].x,points[i].y);
        glEnd();
    }
}

void drawRect(float2 pos, float2 dim, float shade)
{
    glColor3f(0.31*shade, 0.176*shade, 0.5*shade);
    glBegin(GL_POLYGON);
    glVertex2f(pos.x, pos.y);
    glVertex2f(pos.x+dim.x, pos.y);
    glVertex2f(pos.x+dim.x, pos.y+dim.y);
    glVertex2f(pos.x, pos.y+dim.y);
    glEnd();
}

void drawCircle(float2 center, float radius, int segments, float thickness, float color[3])
{
    double pi = 3.14159265358979323846;

    (color==NULL) ? glColor3f(1.0, 1.0, 1.0):glColor3f(color[0], color[1], color[2]);
    glLineWidth(thickness);
    glBegin(GL_LINE_LOOP);
    for(int i = 0; i <= segments; i++)
    {
        float theta = i*2.0*pi/segments;
        glVertex2f(radius*cos(theta)+center.x, radius*sin(theta)+center.y);
    }
    glEnd();
}

void drawText(float x, float y, char string[MAX_STRING])
{
    // GLFW doesn't have built-in text rendering like GLUT
    // For now, print to console - you can implement proper text rendering later
    printf("Text at (%.2f, %.2f): %s\n", x, y, string);
    
    // Alternative: Simple line-based text rendering using OpenGL primitives
    // This is a very basic implementation - for production use a proper text library
    /*
    glPushMatrix();
    glTranslatef(x, y, 0);
    glScalef(0.0001, 0.0001, 0);
    // Draw simple lines to form letters - very basic implementation
    // You would need to implement each letter as line segments
    glPopMatrix();
    */
}

void drawDensity(int *density, float2 *densityCenters, float *range, int b, int n)
{
    for(int i = 0; i < b*b; i++)
    {
        if(density[i])
        {
            float xr[2] = {range[i%b], range[i%b+1]};
            float yr[2] = {range[b-1-i/b], range[b-1-i/b+1]};

            drawRect(make_float2(xr[0], yr[0]), make_float2(xr[1]-xr[0], yr[1]-yr[0]), ((float)density[i])/sqrtf(n));
        }
    }
    
    for(int i = 0; i < b*b; i++)
    {
        if(density[i])
        {
            float color[] = {0.3, 0.3, 1.0}; 
            drawPoint(densityCenters[i], 2.0, color);
        }
    }
}

void drawBubbles(Wall bubbles[B*B])
{
    for(int i = 0; i < B*B;i++)
    {
        if(bubbles[i].strength != 0) drawCircle(bubbles[i].center, bubbles[i].radius*scale, 100, bubbles[i].thickness, Blue);
    }
}

void info()
{
    glfwMakeContextCurrent(window1);
    glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(1.0, 1.0, 1.0);
    
    for(int i = 0;i<N;i++)
    {
        char s[MAX_STRING];
        sprintf(s, "node %d: %f", i, mag(rs.nodes[i].pos));
        float y = -1.8*((float)i/(float)N) + 0.9;
        drawText(-0.9, y, s);
    }

    char pressureText[MAX_STRING];
    sprintf(pressureText, "pressure: %f", rs.pressure);
    drawText(-0.9, -0.9, pressureText);

    glfwSwapBuffers(window1);
}

void render()
{
    // glfwMakeContextCurrent(window0);
    // glClear(GL_COLOR_BUFFER_BIT);
    // getNodePositions(p, rs.nodes, N);
    // linearScalePoints(p, N, scale);
    // linspace(rs.range, -scale, scale, B+1, 1);
    // linearScalePoints(rs.densityCenters, B*B, scale);
    // drawDensity(rs.densities, rs.densityCenters, rs.range, B, rs.numberOfNodes);

    // float2 center = {0,0};
    // float innerCircleColor[3];
    // float outerCircleColor[3];
    // if(rs.innerWall.direction){ for(int i = 0;i<3;i++) innerCircleColor[i] = Green[i];}
    // else { for(int i = 0;i<3;i++) innerCircleColor[i] = LightGray2[i]; }

    // switch(rs.outerWall.direction)
    // {
    //     case -1:
    //         for(int i = 0;i<3;i++) outerCircleColor[i] = Red[i];
    //         break;
    //     case 0:
    //         for(int i = 0;i<3;i++) outerCircleColor[i] = LightGray2[i];
    //         break;
    //     case 1:
    //         for(int i = 0;i<3;i++) outerCircleColor[i] = Green[i];
    //         break;
    //     default:
    //         for(int i = 0;i<3;i++) outerCircleColor[i] = Green[i];
    //         break;
    // }
    // drawCircle(center, rs.outerWall.radius*scale, 100, rs.outerWall.thickness, outerCircleColor);
    // drawCircle(center, rs.innerWall.radius*scale, 100, rs.innerWall.thickness, innerCircleColor);

    // for(int i = 0; i < B*B;i++)
    // {
    //     linearScalePoint(&rs.bubbles[i].center, scale);
    // }
    // drawBubbles(rs.bubbles);
    // drawPoints(p, N, 10.0, LightGray0);
    // glfwSwapBuffers(window0);
    glfwMakeContextCurrent(window0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Simple test drawing instead of complex logic
    glColor3f(1.0, 0.0, 0.0); // Red
    glPointSize(20.0);
    glBegin(GL_POINTS);
    glVertex2f(0.0f, 0.0f);    // Center point
    glVertex2f(0.5f, 0.5f);    // Top-right
    glVertex2f(-0.5f, -0.5f);  // Bottom-left
    glEnd();
    
    // Test circle
    glColor3f(0.0, 1.0, 0.0); // Green
    glBegin(GL_LINE_LOOP);
    for(int i = 0; i < 32; i++) {
        float theta = i * 2.0f * 3.14159f / 32.0f;
        glVertex2f(0.3f * cos(theta), 0.3f * sin(theta));
    }
    glEnd();
    
    glfwSwapBuffers(window0);
}

void drawPath()
{
    glfwMakeContextCurrent(window0);
    glClear(GL_COLOR_BUFFER_BIT);
    getNodeInitPositions(p, rs.nodes, N);
    linearScalePoints(p, N, scale);

    glLineWidth(2.0);
    glColor3f(1.0, 0.3, 0.3);
    glBegin(GL_LINE_LOOP);
    for(int i = 0; i < N;i++)
    {
        glVertex2f(p[rs.nBodyPath[i]].x, p[rs.nBodyPath[i]].y);
    }
    glVertex2f(p[rs.nBodyPath[0]].x, p[rs.nBodyPath[0]].y);
    glEnd();
    drawPoints(p, N, 10.0, LightGray0);
    glfwSwapBuffers(window0);
}

void nBodyRenderInit(int argc, char** argv, RunState runState)
{
    rs = runState;

    // Set error callback
    glfwSetErrorCallback(error_callback);
    
    // Initialize GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return;
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, 4); // Multisampling
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    // Create main visualization window
    window0 = glfwCreateWindow(DIM, DIM, "Traveling Salesman Problem", NULL, NULL);
    if (!window0)
    {
        fprintf(stderr, "Failed to create main window\n");
        glfwTerminate();
        return;
    }
    
    // Set main window position
    glfwSetWindowPos(window0, 100, 100);
    
    // Create info window
    window1 = glfwCreateWindow(256, 768, "Info", NULL, window0); // Share context with main window
    if (!window1)
    {
        fprintf(stderr, "Failed to create info window\n");
        glfwDestroyWindow(window0);
        glfwTerminate();
        return;
    }
    
    // Position info window next to main window
    glfwSetWindowPos(window1, 100 + DIM, 100);
    
    // Set callbacks for both windows
    glfwSetWindowCloseCallback(window0, window_close_callback);
    glfwSetWindowCloseCallback(window1, window_close_callback);
    glfwSetMouseButtonCallback(window0, mouse_button_callback);
    
    // Initialize OpenGL context for main window
    glfwMakeContextCurrent(window0);
    
    // Initialize GLEW
    GLenum glewError = glewInit();
    if (glewError != GLEW_OK)
    {
        fprintf(stderr, "Failed to initialize GLEW: %s\n", glewGetErrorString(glewError));
        glfwTerminate();
        return;
    }
    
    // Configure main window
    glClearColor(0.1725490196, 0.1725490196, 0.1725490196, 1.0);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    
    // Configure info window
    glfwMakeContextCurrent(window1);
    glClearColor(0.1725, 0.1725, 0.1725, 1.0);
    
    // Enable V-Sync
    glfwSwapInterval(1);
}

void nBodyRenderUpdate(RunState runState)
{
    rs = runState;
    render();
    glfwPollEvents();
    info();
    glfwPollEvents();
}

void nBodyDrawPath(RunState runState)
{
    rs = runState;
    drawPath();
    glfwPollEvents();
}

bool waitForMouseClick()
{
    glfwPollEvents();
    return !clicked;
}

int getWindowState()
{ 
    return (closed || glfwWindowShouldClose(window0) || glfwWindowShouldClose(window1)) ? 0 : 1;
}

int hasMouseClicked()
{ 
    return clicked;
}

void cleanupRendering()
{
    if (window0) glfwDestroyWindow(window0);
    if (window1) glfwDestroyWindow(window1);
    glfwTerminate();
}