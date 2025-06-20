import numpy as np
from scipy.linalg import block_diag

class Kalman:
    #TODO: add option for numba implementation

    def __init__(self, dt, P_mul, R_mul, Q_mul):

        self.dt = dt
        self.P_mul = P_mul
        self.R_mul = R_mul
        self.Q_mul = Q_mul

    def __init_matrices(self, init_X):

        self.ms_size = init_X.shape[0]
        self.full_size = self.ms_size * 3

        self.A = self.__get_A_matrix(self.dt)
        self.B = np.zeros((self.full_size, self.ms_size), dtype=np.float64, order='C')
        self.X = np.zeros((self.full_size), dtype=np.float64, order='C')
        self.X[0::3] = init_X

        tmp_p_mat = np.ones((3,3), dtype=np.float64, order='C')
        repeated_p_mat = self.ms_size * [tmp_p_mat]
        self.P = np.eye(self.full_size, dtype=np.float64, order='C') * self.P_mul#(block_diag(*repeated_p_mat) * self.P_mul)

        self.U = np.zeros(self.ms_size,dtype=np.float64, order='C')
        self.Q = self.__get_Q_matrix(self.dt)
        self.Q *= self.Q_mul

        tmp_h_row = np.array([1, 0, 0], dtype=np.float64, order='C')
        repeated_arr = self.ms_size * [tmp_h_row]
        self.H = block_diag(*repeated_arr)

        self.R = np.eye(self.ms_size, dtype=np.float64, order='C')
        self.R *= self.R_mul

    def __get_A_matrix(self, dt):
        tmp_mat = np.eye(3, dtype=np.float64, order='C')
        tmp_mat[0, 1] = dt
        tmp_mat[1, 2] = dt
        tmp_mat[0, 2] = 0.5 * dt ** 2

        repeated_arr = self.ms_size * [tmp_mat]

        transi_matrix = block_diag(*repeated_arr)

        return transi_matrix

    def __get_Q_matrix(self, dt):
        tmp_mat = np.eye(3, dtype=np.float64, order='C')

        tmp_mat[0, 0] = 1/20 * (dt ** 5)
        tmp_mat[0, 1] = 1/8 * (dt ** 4)
        tmp_mat[1, 0] = 1/8 * (dt ** 4)
        tmp_mat[2, 0] = 1/6 * (dt ** 3)
        tmp_mat[1, 1] = 1/3 * (dt ** 3)
        tmp_mat[0, 2] = 1/6 * (dt ** 3)
        tmp_mat[2, 1] = 1/2 * (dt ** 2)
        tmp_mat[1, 2] = 1/2 * (dt ** 2)
        tmp_mat[2, 2] = dt

        repeated_arr = self.ms_size * [tmp_mat]
        transi_matrix = block_diag(*repeated_arr)

        return transi_matrix 


    def __predict(self, update=False):
        X_temp = np.dot(self.A, self.X) + np.dot(self.B, self.U)
        P_temp = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

        if(update):
            self.X = X_temp
            self.P = P_temp

        return X_temp[0::3]

    def __update(self, Y, Y_sec = None, R_sec = None):
        

        if(Y_sec is not None):
            R_sec = R_sec + self.R
            #R_sec = np.concatenate([self.R[None, :], self.R[None,  :]])
            R_sec_inv = np.linalg.inv(R_sec)

            R_inv = np.linalg.inv(self.R)

            mean_weighted_R = R_inv + np.sum(R_sec_inv, axis=0)
            mean_weighted_R_inv = np.linalg.inv(mean_weighted_R)

            R_tmp = mean_weighted_R_inv
            #mean_H = mean_weighted_R_inv @ (mean_weighted_R @ self.H)
            #H_tmp = R_tmp @ (np.sum(R_sec_inv @ self.H, axis=0) + R_inv @ self.H)
            H_tmp = self.H
            Y_tmp = R_tmp @ (np.sum(np.einsum("nki, ni -> nk", R_sec_inv, Y_sec), axis=0) + R_inv @ Y)

        else:
            H_tmp = self.H
            R_tmp = self.R
            Y_tmp = Y

        self.IM = np.dot(H_tmp, self.X)
        self.IS = R_tmp + np.dot(H_tmp, np.dot(self.P, H_tmp.T))
        i, j = np.nonzero(self.IS)
        self.K = np.dot(self.P, np.dot(H_tmp.T, np.linalg.inv(self.IS))) #Gain
        
        self.IN = Y_tmp - self.IM #innovation
        self.X = self.X + np.dot(self.K, self.IN)        
        self.P = self.P - np.dot(self.K, np.dot(self.IS, self.K.T))


    #TODO: Fix for initial vector (what if it is zero)
    def filter_old(self, seq, R_mat_sec=None):
        #Needs to be in shape (Steps * Feats)
        seq_aux = None

        if(seq.ndim == 3): #TODO:For more than 3 views?
            assert(R_mat_sec is not None)
            assert(R_mat_sec.shape == (seq.ndim - 1, seq.shape[-1], seq.shape[-1]))
            seq_aux = seq[1:seq.shape[0]]
            seq = seq[0]

        self.__init_matrices(seq[0])

        seq_new = np.copy(seq)
        seq_new[0, :] = seq[0, :]

        for i in range(0, seq.shape[0]):
            
            if(i > 0):
                res_pred = self.__predict(True)
                seq_new[i, :] = res_pred
            
            param_sec = None
            if(seq_aux is not None and i > 0):
                param_sec = seq_aux[:,i]
            self.__update(seq[i], param_sec, R_mat_sec)
                        
        return seq_new




    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        #pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, P):

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        P_tmp = (lambda_ + n)*P
        U_tmp, S_t, _ = np.linalg.svd(P_tmp)
        sqrt = U_tmp @ np.diag(np.sqrt(S_t)) @ U_tmp.T

        sigmas = np.zeros((2*n+1, n))
        # sigmas[0] = x
        # for k in range(n):
        #     sigmas[k+1]   = self.subtract(x, -U[k])
        #     sigmas[n+k+1] = self.subtract(x, U[k])

        sigmas[0] = x
        sigmas[1:n+1] = self.subtract(x, -sqrt[0:n])
        sigmas[n+1:n+n+1] = self.subtract(x, sqrt[0:n])

        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.

        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)