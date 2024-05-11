#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include "vector_op.h"

void print_mat_to_file(double * mat, size_t rows, size_t cols, char * filename)
{
    FILE *fptr;
    size_t i, j;
    char file_dir[100];
    // Open a file in append mode
    
    strcpy(file_dir,"files//");
    strcat(file_dir,filename);
    strcat(file_dir,".txt");
    fptr = fopen(file_dir, "a+");

    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < cols; j++)
        {
            fprintf(fptr, "%.6f ", mat[i * cols + j]);
        }
        fprintf(fptr, "\n");
    } 
    fprintf(fptr, "\n");
    // Close the file
    fclose(fptr);
}

size_t inv_mat_blas(double * A, size_t v, double * A_INV)
{

    int N = v;
    int LWORK = (int)(1.01 * N*N);
    int INFO;
    double * WORK = malloc(sizeof(double) * LWORK);
    int * IPIV = malloc(sizeof(int) * N);
    int ret;

    memcpy(A_INV, A, sizeof(double) * v * v);

    dgetrf_(&N,&N,A_INV,&N,IPIV,&INFO);
    if(INFO != 0)
    {
        free(IPIV);
        free(WORK);
        printf("dgetrf returned %d \n", ret);
        return INFO;
    }

    dgetri_(&N,A_INV,&N,IPIV,WORK,&LWORK,&INFO);

    if(INFO != 0)
    {
        free(IPIV);
        free(WORK);
        printf("dgetrf returned %d \n", ret);
        return INFO;
    }

    free(IPIV);
    free(WORK);

    return INFO;
}

void add_to_vec(double * A, double * B, size_t v)
{   
    size_t i;
    for (i = 0; i < v; ++i)
        A[i] = A[i] + B[i];
}

void sub_from_vec(double * A, double * B, size_t v)
{   
    size_t i;
    for (i = 0; i < v; ++i)
        A[i] = A[i] - B[i];
}

void sub_from_mat(double * A, double * B,  size_t rows, size_t cols)
{   
    size_t i, j;
    for(i = 0; i < rows; i++)
    {
        for(j = 0; j < cols; j++)
        {   
            A[i * cols + j] = A[i * cols + j] - B[i * cols + j];
        }
    }
}



