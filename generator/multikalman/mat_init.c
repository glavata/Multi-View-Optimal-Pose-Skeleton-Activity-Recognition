#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mat_init.h"

void free_mat_struct(mat_p * mat)
{   
    free(mat->A.data);
    free(mat->B.data);
    free(mat->X.data);
    free(mat->P.data);
    free(mat->U.data);
    free(mat->Q.data);
    free(mat->H.data);
    free(mat->R.data);
}

mat init_mat_ptr(double * ptr, size_t rows, size_t cols)
{
    mat res_mat;
    res_mat.data = ptr;
    res_mat.rows = rows;
    res_mat.cols = cols;

    return res_mat;
}

vec init_vec_ptr(double * ptr, size_t len)
{
    vec res_vec;
    res_vec.data = ptr;
    res_vec.len = len;

    return res_vec;
}

mat init_zeros_mat(size_t rows, size_t cols)
{   
    mat res_mat;
    double * zero_mat = malloc(sizeof(double) * rows * cols);
    memset(zero_mat, 0, rows * cols * sizeof(double));
    return init_mat_ptr(zero_mat, rows, cols);
}

vec init_zeros_vec(size_t len)
{   
    vec res_vec;
    double * zero_vec = malloc(sizeof(double) * len);
    memset(zero_vec, 0, len * sizeof(double));
    return init_vec_ptr(zero_vec, len);
}

mat init_A(size_t order, double dt)
{
    //TODO: change hardcoded 3
    size_t i;
    size_t full_size = order * 3; 
    mat A_res = init_zeros_mat(full_size, full_size);
   
    for(i = 0; i < full_size; i++)
    {
        A_res.data[i * full_size + i] = 1;
        if(i % 3 == 0)
        {
            A_res.data[i * full_size + i + 1] = dt;
            A_res.data[i * full_size + i + 2] = 0.5l * dt * dt;
            A_res.data[(i + 1) * full_size + i + 2] = dt;
        }
    }

    return A_res;
}

mat init_Q(size_t order, double dt, double Q_mul)
{
    //TODO: change hardcoded 3
    size_t i;
    size_t full_size = order * 3; 
    mat Q_res = init_zeros_mat(full_size, full_size);
    
    for(i = 0; i < full_size; i++)
    {
        if(i % 3 == 0)
        {
            Q_res.data[i * full_size + i] = 0.25l * pow(dt, 4) * Q_mul;
            Q_res.data[i * full_size + i + 1] = 0.5l * pow(dt, 3) * Q_mul;
            Q_res.data[(i + 1) * full_size + i] = 0.5l * pow(dt, 3) * Q_mul;
            Q_res.data[(i + 2) * full_size + i] = 0.5l * pow(dt, 2) * Q_mul;
            Q_res.data[(i + 1) * full_size + i + 1] = dt * dt * Q_mul;
            Q_res.data[i * full_size + i + 2] = 0.5l * pow(dt, 2) * Q_mul;
            Q_res.data[(i + 2)* full_size + i + 1] = dt * Q_mul;
            Q_res.data[(i + 1)* full_size + i + 2] = dt * Q_mul;
            Q_res.data[(i + 2)* full_size + i + 2] = 1.0l * Q_mul;

        }
    }

    return Q_res;
}

mat init_R(size_t order, double R_mul)
{
    //TODO: change hardcoded 3
    size_t i;
    mat R_res = init_zeros_mat(order, order);
    
    for(i = 0; i < order; i++)
    {
        R_res.data[i * order + i] = 1.0l * R_mul;
    }
    return R_res;
}

mat init_P(size_t order, double P_mul)
{
    //TODO: change hardcoded 3
    size_t i, j, k;
    size_t full_size = order * 3; 
    mat P_res = init_zeros_mat(full_size, full_size);
    
    for(i = 0; i < order; i++)
    {
        for(j = i * 3; j < i * 3 + 3; j++)
        {
            for(k = i * 3; k < i * 3 + 3; k++)
            {
                P_res.data[j * full_size + k] = 1.0l * P_mul;
            }
        }
    }

    return P_res;
}

mat init_H(size_t order)
{
    //TODO: change hardcoded 3
    size_t j;
    mat H_res = init_zeros_mat(order, order * 3);
    
    for(j = 0; j < order; j++)
    {
        H_res.data[j * (order * 3) + (j * 3)] = 1.0l;
    }

    return H_res;
}


vec init_X(double * X_init, size_t order)
{
    //TODO: change hardcoded 3
    size_t i;
    size_t full_size = order * 3; 
    vec X_res = init_zeros_vec(full_size);

    for(i = 0; i < order; i++)
    {
        X_res.data[i * 3] = X_init[i];
    }

    return X_res;
}

mat_p * init_matrices(double * X_init, size_t len, double ms, double P_mul, double R_mul, double Q_mul)
{
    size_t full_len = len * 3;

    mat A_mat = init_A(len, ms);
    mat B_mat = init_zeros_mat(full_len, len);
    vec X_vec = init_X(X_init, len);
    mat P_mat = init_P(len, P_mul);
    vec U_vec = init_zeros_vec(len);
    mat Q_mat = init_Q(len, ms, Q_mul);
    mat H_mat = init_H(len);
    mat R_mat = init_R(len, R_mul);

    mat_p * M = malloc(sizeof(mat_p));
    *M = (mat_p){A_mat, B_mat, P_mat, Q_mat, H_mat, R_mat, X_vec, U_vec};

    return M;
}

mat_p * init_matrices_pnt(np_p mat_ptrs, size_t full_len, size_t len)
{
    mat A_mat = init_mat_ptr(mat_ptrs.A, full_len, full_len);
    mat B_mat = init_mat_ptr(mat_ptrs.B, full_len, len);
    vec X_vec = init_vec_ptr(mat_ptrs.X, full_len);
    mat P_mat = init_mat_ptr(mat_ptrs.P, full_len, full_len);
    vec U_vec = init_vec_ptr(mat_ptrs.U, len);
    mat Q_mat = init_mat_ptr(mat_ptrs.Q, full_len, full_len);
    mat H_mat = init_mat_ptr(mat_ptrs.H, len, full_len);
    mat R_mat = init_mat_ptr(mat_ptrs.R, len, len);

    mat_p * M = malloc(sizeof(mat_p));
    *M = (mat_p){A_mat, B_mat, P_mat, Q_mat, H_mat, R_mat, X_vec, U_vec};

    return M;
}