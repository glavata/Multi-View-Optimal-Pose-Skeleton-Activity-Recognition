typedef struct matrix_s 
{
    double * data;
    size_t rows;
    size_t cols;
} mat;

typedef struct vector_s
{
    double * data;
    size_t len;
} vec;

typedef struct mat_pointers {
    mat A, B, P, Q, H, R;
    vec X, U;
} mat_p;

typedef struct numpy_pointers{
    double * A, *B, *P, *Q, *H, *R, *X, *U;
} np_p;

mat_p * init_matrices(double * X_init, size_t len, double ms, double P_mul, double R_mul, double Q_mul);
mat_p * init_matrices_pnt(np_p mat_ptrs, size_t full_len, size_t len);
void free_mat_struct(mat_p * mat);

mat init_zeros_mat(size_t rows, size_t cols);
vec init_zeros_vec(size_t len);
vec init_X(double * X_init, size_t order);
mat init_H(size_t order);
mat init_P(size_t order, double P_mul);
mat init_R(size_t order, double R_mul);
mat init_Q(size_t order, double dt, double Q_mul);
mat init_A(size_t order, double dt);