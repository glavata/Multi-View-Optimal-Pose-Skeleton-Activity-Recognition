

void dgetrf_ (int * m, int * n, double * A, int * LDA, int * IPIV, int * INFO);
void dgetri_ (int * n, double * A, int * LDA, int * IPIV,double * WORK, int * LWORK, int * INFO);

void print_mat_to_file(double * mat, size_t rows, size_t cols, char * filename);

size_t inv_mat_blas(double * A, size_t v, double * A_inv);
void add_to_vec(double * A, double * B, size_t v);
void sub_from_vec(double * A, double * B, size_t v);
void sub_from_mat(double * A, double * B,  size_t rows, size_t cols);

