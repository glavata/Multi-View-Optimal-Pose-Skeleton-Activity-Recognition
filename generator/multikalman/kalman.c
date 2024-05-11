
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

#include "mat_init.h"
#include "vector_op.h"


#ifdef _WIN32
#   define API __declspec(dllexport)
#else
#   define API
#endif


double * predict_step(mat_p * m, size_t full_len, size_t len)
{
    size_t i;

    double * X_A = malloc(sizeof(double) * full_len);

    double * B_U = malloc(sizeof(double) * full_len);

    double * P_A_tr = malloc(sizeof(double) * full_len * full_len);

    double * A_P_A = malloc(sizeof(double) * full_len * full_len);

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m->A.rows, m->A.cols, 1, m->A.data, m->A.cols, m->X.data, 1, 0, X_A, 1);

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m->B.rows, m->B.cols, 1, m->B.data, m->B.cols, m->U.data, 1, 0, B_U, 1);

    add_to_vec(X_A, B_U, full_len);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m->P.rows, m->A.rows, m->P.cols, 1, m->P.data, m->P.cols, m->A.data, m->A.cols, 0, P_A_tr, full_len);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m->A.rows, full_len, m->A.cols, 1, m->A.data, m->A.cols, P_A_tr, full_len, 0, A_P_A, full_len);

    add_to_vec(A_P_A, m->Q.data, full_len * full_len); //Instead of sum_mat_2d

    double * X_tmp_x = malloc(sizeof(double) * len);

    for(i = 0; i < len; i++)
    {   
        X_tmp_x[i] = X_A[i * 3];
        m->X.data[i * 3] = X_A[i * 3];
        m->X.data[i * 3 + 1] = X_A[i * 3 + 1];
        m->X.data[i * 3 + 2] = X_A[i * 3 + 2];
    }

    memcpy(m->P.data, A_P_A, full_len * full_len * sizeof(double));

    free(X_A); free(B_U); free(P_A_tr); free(A_P_A);

    return X_tmp_x;
}

size_t update_step(double * Y, mat_p * m, size_t full_len, size_t len)
{

    //lda ldb ldc remain the same -> sec dim of matrix
    //m n k change depending on trans or notrans
    size_t res_inv_f;

    double * I_M = malloc(sizeof(double) * len);

    double * P_H_tr = malloc(sizeof(double) * full_len * len);

    double * H_P_H_tr = malloc(sizeof(double) * len * len); //I_S
    
    double * I_N = malloc(sizeof(double) * len);

    double * H_tr_I_S_inv = malloc(sizeof(double) * full_len * len);

    double * K = malloc(sizeof(double) * full_len * len);

    double * K_IN = malloc(sizeof(double) * full_len);

    double * IS_K_tr = malloc(sizeof(double) * len * full_len);

    double * K_IS_K_tr = malloc(sizeof(double) * full_len * full_len);

    double * I_S_inv = malloc(sizeof(double) * len * len);

    cblas_dgemv(CblasRowMajor, CblasNoTrans, m->H.rows, m->H.cols, 1, m->H.data, m->H.cols, m->X.data, 1, 0, I_M, 1);

    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m->P.rows, m->H.cols, m->P.cols, 1, m->P.data, m->P.cols, m->H.data, m->H.cols, 0, P_H_tr, len);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m->P.rows, m->H.rows, m->P.cols, 1, m->P.data, m->P.cols, m->H.data, m->H.cols, 0, P_H_tr, len);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m->H.rows, len, m->H.cols, 1, m->H.data, m->H.cols, P_H_tr, len, 0, H_P_H_tr, len);

    add_to_vec(H_P_H_tr, m->R.data, len * len); //Instead of sum_mat_2d

    memcpy(I_N, Y, sizeof(double) * len);
    sub_from_vec(I_N, I_M, len);

    res_inv_f = inv_mat_blas(H_P_H_tr, len, I_S_inv); //I_S_inv

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m->H.cols, len, m->H.rows, 1, m->H.data, m->H.cols, I_S_inv, len, 0, H_tr_I_S_inv, len);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m->P.rows, len, m->P.cols, 1, m->P.data, m->P.cols, H_tr_I_S_inv, len, 0, K, len);

    cblas_dgemv(CblasRowMajor, CblasNoTrans, full_len, len, 1, K, len, I_N, 1, 0, K_IN, 1);

    add_to_vec(m->X.data, K_IN, full_len);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, len, full_len, len, 1, H_P_H_tr, len, K, len, 0, IS_K_tr, full_len);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, full_len, full_len, len, 1, K, len, IS_K_tr, full_len, 0, K_IS_K_tr, full_len);

    sub_from_mat(m->P.data, K_IS_K_tr, full_len, full_len);

    free(I_S_inv);
    free(I_M); free(P_H_tr); free(H_P_H_tr);
    free(I_N); free(H_tr_I_S_inv); free(K); 
    free(K_IN); free(IS_K_tr); free(K_IS_K_tr);

    return res_inv_f;
}


double * filter_loop(double * mat, mat_p * matrices, size_t rows, size_t cols, size_t full_len)
{
    //cols == len
    size_t i, res_u;
    double * tmp_row;
    double * res_mat = malloc(sizeof(double) * rows * cols);
    memcpy(res_mat, mat, cols * sizeof(double));

    for(i = 1; i < rows; i++)
    {
        tmp_row = predict_step(matrices, full_len, cols);
        memcpy(res_mat + i * cols, tmp_row, cols * sizeof(double));
        free(tmp_row);
        res_u = update_step(mat + i * cols, matrices, full_len, cols);
        if(res_u)
        {
            printf("Error during update step, skipping \n");
        }
    }
    return res_mat;
}


API double * filter_init(double *mat, const size_t rows, const size_t cols, np_p mats) 
{
    size_t len = cols;
    size_t full_len = len * 3;
    double * res_mat;

    mat_p * matrices = init_matrices_pnt(mats, full_len, len);
    res_mat = filter_loop(mat, matrices, rows, cols, full_len);

    free_mat_struct(matrices);
    return res_mat;
}


API double * filter_noinit(double * mat, const size_t rows, const size_t cols, const double * params) 
{
    double ms, P_mul, R_mul, Q_mul;
    size_t len = cols;
    size_t full_len = len * 3;
    double * res_mat;

    ms = params[0];
    P_mul = params[1];
    R_mul = params[2];
    Q_mul = params[3];

    mat_p * matrices = init_matrices(mat, len, ms, P_mul, R_mul, Q_mul);
    res_mat = filter_loop(mat, matrices, rows, cols, full_len);

    free_mat_struct(matrices);
    free(matrices);

    return res_mat;
}


int main(int argc, char *argv[])
{
    return 0;
}