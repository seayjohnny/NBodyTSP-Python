#ifndef DATAIOH
#define DATAIOH

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <stdio.h>

struct OutputParametersSpring
{
    double slopeRepulsion;
    double forceCutoff;
    double magAttraction;
    double wallStrength;
    double damp;
    double mass;
    int bins;
    int minBinDensity;
    int lowerPressureLimit;
    int upperPressureLimit;
    double optimalCost;
    double nBodyCost;
    double percentDiff;
    int drawn;
    double runTime;
};

struct OutputParametersLJ
{
    double m;
    double p;
    double q;
    double wallStrength;
    double damp;
    double mass;
    int bins;
    int minBinDensity;
    int lowerPressureLimit;
    int upperPressureLimit;
    double optimalCost;
    double nBodyCost;
    double percentDiff;
    int drawn;
    double runTime;
};

#define MAX_FILE_NAME 20

int getNumberOfLines(const char filename[MAX_FILE_NAME])
{
	FILE *fp = fopen(filename, "r");
    if (!fp) return 0; // Add error checking
    
    int count = 1;
    for(char c = getc(fp); c != EOF; c = getc(fp))
    {
        if(c == '\n') count += 1;
    }

    fclose(fp);
    return count;
}

void loadNodesFromFile(const char filename[MAX_FILE_NAME], float2 *nodes)
{   
    FILE *fp = fopen(filename, "r");
    if (!fp) return; // Add error checking
    
    int i = 0;
    while(1)
    {
        fscanf(fp,"%f %f", &nodes[i].x, &nodes[i].y );
        i += 1;
        if(feof(fp))
        {
            break;
        }
    }
    fclose(fp);
}

void loadOptimalPath(const char filename[MAX_FILE_NAME], int *path, int n)
{
    FILE *fp = fopen(filename, "r");
    if (!fp) return; // Add error checking
    
    int i = 0;
    while(i < n)
    {
        fscanf(fp,"%d", &path[i]);
        path[i]--;
        i += 1;
        if(feof(fp))
        {
            break;
        }
    }
    fclose(fp);
}

void logSpringRun(const char filename[MAX_FILE_NAME], OutputParametersSpring params)
{
    FILE *fp = fopen(filename, "a");
    if (!fp) return; // Add error checking
    
    fprintf(fp, "=================================\n");
    fprintf(fp, "\tSlope Repulsion: %f\n", params.slopeRepulsion);
    fprintf(fp, "\tForce Cutoff: %f\n", params.forceCutoff);
    fprintf(fp, "\tMag Attraction: %f\n", params.magAttraction);
    fprintf(fp, "\tWall Strength: %f\n", params.wallStrength);
    fprintf(fp, "\tDamp: %f\n", params.damp);
    fprintf(fp, "\tMass: %f\n\n", params.mass);
    fprintf(fp, "\tBins: %d\n", params.bins);
    fprintf(fp, "\tMin Bin Density: %d\n\n", params.minBinDensity);
    fprintf(fp, "\tLower Pressure Limit: %d\n", params.lowerPressureLimit);
    fprintf(fp, "\tUpper Pressure Limit: %d\n\n", params.upperPressureLimit);
    fprintf(fp, "\tOptimal Cost: %f\n", params.optimalCost);
    fprintf(fp, "\tN-Body Cost: %f\n", params.nBodyCost);
    fprintf(fp, "\tPercent Diff: %f\n", params.percentDiff);
    fprintf(fp, "\tDrawn: %s\n", params.drawn ? "Yes":"No");
    fprintf(fp, "\tRun Time: %.4f s\n", params.runTime/1000000);
    fprintf(fp, "=================================\n");
    fclose(fp);
}

void logLJRun(const char filename[MAX_FILE_NAME], OutputParametersLJ params)
{
    FILE *fp = fopen(filename, "a");
    if (!fp) return; // Add error checking
    
    fprintf(fp, "=================================\n");
    fprintf(fp, "\tM: %f\n", params.m);
    fprintf(fp, "\tp: %f\n", params.p);
    fprintf(fp, "\tq: %f\n", params.q);
    fprintf(fp, "\tWall Strength: %f\n", params.wallStrength);
    fprintf(fp, "\tDamp: %f\n", params.damp);
    fprintf(fp, "\tMass: %f\n\n", params.mass);
    fprintf(fp, "\tBins: %d\n", params.bins);
    fprintf(fp, "\tMin Bin Density: %d\n\n", params.minBinDensity);
    fprintf(fp, "\tLower Pressure Limit: %d\n", params.lowerPressureLimit);
    fprintf(fp, "\tUpper Pressure Limit: %d\n\n", params.upperPressureLimit);
    fprintf(fp, "\tOptimal Cost: %f\n", params.optimalCost);
    fprintf(fp, "\tN-Body Cost: %f\n", params.nBodyCost);
    fprintf(fp, "\tPercent Diff: %f\n", params.percentDiff);
    fprintf(fp, "\tDrawn: %s\n", params.drawn ? "Yes":"No");
    fprintf(fp, "\tRun Time: %.4f s\n", params.runTime/1000000);
    fprintf(fp, "=================================\n");
    fclose(fp);
}

void logLJRunShort(const char filename[MAX_FILE_NAME], OutputParametersLJ params)
{
    FILE *fp = fopen(filename, "a");
    if (!fp) return; // Add error checking
    
    fprintf(fp, "=================================\n");
    fprintf(fp, "\tM: %f\n", params.m);
    fprintf(fp, "\tp: %f\n", params.p);
    fprintf(fp, "\tq: %f\n", params.q);

    fprintf(fp, "\tOptimal Cost: %f\n", params.optimalCost);
    fprintf(fp, "\tN-Body Cost: %f\n", params.nBodyCost);
    fprintf(fp, "\tPercent Diff: %f\n", params.percentDiff);

    fprintf(fp, "\tRun Time: %.4f s\n", params.runTime/1000000);
    fprintf(fp, "=================================\n");
    fclose(fp);
}

// Platform-specific timing functions
#ifdef _WIN32
// Windows implementation using QueryPerformanceCounter
static LARGE_INTEGER frequency;
static bool frequency_initialized = false;

void initTimer() 
{
    if (!frequency_initialized) {
        QueryPerformanceFrequency(&frequency);
        frequency_initialized = true;
    }
}

void startTimer(double *timer)
{
    initTimer();
    LARGE_INTEGER temp;
    QueryPerformanceCounter(&temp);
    // Convert to microseconds for compatibility with original code
    *timer = (double)(temp.QuadPart * 1000000) / frequency.QuadPart;
}

double getTimer(double *timer)
{
    initTimer();
    LARGE_INTEGER temp;
    QueryPerformanceCounter(&temp);
    double current_time = (double)(temp.QuadPart * 1000000) / frequency.QuadPart;
    return current_time - *timer;
}

void endTimer(double *timer)
{
    initTimer();
    LARGE_INTEGER temp;
    QueryPerformanceCounter(&temp);
    double current_time = (double)(temp.QuadPart * 1000000) / frequency.QuadPart;
    *timer = current_time - *timer;
}

#else
// POSIX implementation using gettimeofday
void startTimer(double *timer)
{
    timeval temp;
    gettimeofday(&temp, NULL);
    *timer = (double)(temp.tv_sec * 1000000 + temp.tv_usec);
}

double getTimer(double *timer)
{
    timeval temp;
    gettimeofday(&temp, NULL);
    return((double)(temp.tv_sec * 1000000 + temp.tv_usec) - *timer);
}

void endTimer(double *timer)
{
    timeval temp;
    gettimeofday(&temp, NULL);
    *timer = (double)(temp.tv_sec * 1000000 + temp.tv_usec) - *timer;
}
#endif

#endif