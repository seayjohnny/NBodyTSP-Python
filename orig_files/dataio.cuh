#pragma once

#include <time.h>
#include <stdio.h>
#include "node.cuh"
#include "structs.cuh"

#ifndef MAX_FILENAME
#define MAX_FILENAME 20
#endif

int getNumberOfNodes(const char filename[MAX_FILENAME])
{
    FILE *fp = fopen(filename, "r");
    int count = 1;
    for(char c = getc(fp); c != EOF; c = getc(fp))
    {
        if(c == '\n') count += 1;
    }

    fclose(fp);
    return count;
}

void loadNodes(Node *nodes, const char filename[MAX_FILENAME])
{
    FILE *fp = fopen(filename, "r");
    
    int i = 0;
    while(1)
    {
        fscanf(fp,"%f %f", &nodes[i].initpos.x, &nodes[i].initpos.y );
        nodes[i].id = i;
        nodes[i].pos.x = nodes[i].initpos.x;
        nodes[i].pos.y = nodes[i].initpos.y;
        nodes[i].vel.x = 0.0;
        nodes[i].vel.y = 0.0;
        nodes[i].acc.x = 0.0;
        nodes[i].acc.y = 0.0;
        nodes[i].mass = 1.0;

        i += 1;
        if(feof(fp))
        {
            break;
        }
    }
    fclose(fp);
}

void loadPos(float2* pos, const char filename[MAX_FILENAME])
{
    FILE *fp = fopen(filename, "r");
    
    int i = 0;
    while(1)
    {
        fscanf(fp,"%f %f", &pos[i].x, &pos[i].y );

        i += 1;
        if(feof(fp))
        {
            break;
        }
    }
    fclose(fp);
}

void loadPath(int* path, const char filename[MAX_FILENAME])
{
    FILE *fp = fopen(filename, "r");
    
    int i = 0;
    while(1)
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

void logRun(RunState* rs, const char filename[MAX_FILENAME])
{
    FILE *fp = fopen(filename, "a");
    fprintf(fp, "=================================\n");
    fprintf(fp, "\t(M, p, q): (%f, %f, %f)\n", rs->m, rs->p, rs->q);
    fprintf(fp, "\t(Damp, Mass): (%f, %f)\n", rs->damp, rs->mass);
    fprintf(fp, "\t(LP, UP): (%f, %f)\n", rs->lowerPressureLimit, rs->upperPressureLimit);
    fprintf(fp, "\t(Optimal, N-Body, %%Diff): (%f, %f, %f)\n", rs->optimalCost, rs->nbodyCost, rs->percentDiff);
    fprintf(fp, "\tDrawn: %s\n", rs->drawn ? "Yes":"No");
    fprintf(fp, "\tRun Time: %.4f s\n", rs->runTime/1000000);
    fprintf(fp, "=================================\n");
    fclose(fp);
}

void logCosts(double* results, float4* params, ConstantParameters consts, int b, const char filename[MAX_FILENAME])
{
    FILE *fp = fopen(filename, "a");
    for(int i = 0; i < b; i++)
    {
        fprintf(fp, "%f %f %f %f %f %f %f %f %f %f\n", results[i], params[i].y, params[i].z, params[i].w, consts.damp, consts.mass, consts.wallStrength, consts.forceCutoff, consts.dr, consts.dt);
    }
    fclose(fp);

}

