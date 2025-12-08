#pragma once

// Platform-specific includes
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

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
