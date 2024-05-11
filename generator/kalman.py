import numpy as np
from scipy.linalg import block_diag
from numba import njit
import ctypes
from numpy.ctypeslib import ndpointer

class struct_c_kalman_mat(ctypes.Structure):
    _fields_ = [(key, ctypes.POINTER(ctypes.c_double)) 
                for key in ["A", "B", "P", "Q", "H", "R", "X", "U"]]



class Kalman:
    #TODO: add option for numba implementation

    def __init__(self, ms_size, dt, P_mul, R_mul, Q_mul, init_mat_c=True):
        self.ms_size = ms_size
        self.full_size = ms_size * 3
        self.dt = dt
        self.P_mul = P_mul
        self.R_mul = R_mul
        self.Q_mul = Q_mul

        #self.cur_step = 1
        #self.cur_cov = 0
        #self.first_init = True

        self.init_mat_c = init_mat_c
        self.__init_filter_func()

    def __init_filter_func(self):
        #self.func_f = self.filter_old
        #return
        try:
            lib = ctypes.CDLL("generator/multikalman/kalman.dll", use_last_error=True)
        except:
            self.func_f = self.filter_fast
        else:
            self.func_f = self.filter_c

            openblas_lib = ctypes.cdll.LoadLibrary("generator/multikalman/libopenblas.dll")
            openblas_lib.openblas_set_num_threads(4)

            if(self.init_mat_c):
                fun_tmp = lib.filter_noinit
                fun_tmp.argtypes = [ndpointer(dtype=ctypes.c_double, flags='C_CONTIGUOUS'), 
                ctypes.c_size_t, ctypes.c_size_t, ndpointer(dtype=ctypes.c_double,  flags='C_CONTIGUOUS')]
                fun_tmp.restype = ctypes.POINTER(ctypes.c_double)
            else:
                fun_tmp = lib.filter_init
                fun_tmp.argtypes = [ndpointer(dtype=ctypes.c_double, flags='C_CONTIGUOUS'), 
                ctypes.c_size_t, ctypes.c_size_t, struct_c_kalman_mat]
                fun_tmp.restype = ctypes.POINTER(ctypes.c_double)

            self.fun_ = fun_tmp
            #lib.free_double_p.argtypes = [ctypes.POINTER(ctypes.c_double)]
            #lib.free_double_p.restype = None

            #self.free_fun_ = lib.free_double_p
            
    def get_matrices(self):
        return self.A, self.B, self.P, self.Q, self.H, self.R, self.X, self.U

    def __init_matrices(self, init_X):
        self.A = self.__get_A_matrix(self.dt)
        self.B = np.zeros((self.full_size, self.ms_size), dtype=np.float64, order='C')
        self.X = np.zeros((self.full_size), dtype=np.float64, order='C')
        self.X[0::3] = init_X

        tmp_p_mat = np.ones((3,3), dtype=np.float64, order='C')
        repeated_p_mat = self.ms_size * [tmp_p_mat]
        self.P = (block_diag(*repeated_p_mat) * self.P_mul)

        self.U = np.zeros(self.ms_size,dtype=np.float64, order='C')
        self.Q = self.__get_Q_matrix(self.dt)
        #self.Q = np.eye(self.full_size, dtype=np.float32)
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
        #transi_matrix[diag_ind_dt] = dt
        repeated_arr = self.ms_size * [tmp_mat]

        transi_matrix = block_diag(*repeated_arr)

        return transi_matrix
    
    def __get_Q_matrix(self, dt):
        tmp_mat = np.eye(3, dtype=np.float64, order='C')

        tmp_mat[0, 0] = 0.25 * (dt ** 4)
        tmp_mat[0, 1] = 0.5 * (dt ** 3)
        tmp_mat[1, 0] = 0.5 * (dt ** 3)
        tmp_mat[2, 0] = 0.5 * (dt ** 2)
        tmp_mat[1, 1] = dt ** 2
        tmp_mat[0, 2] = 0.5 * (dt ** 2)
        tmp_mat[2, 1] = dt
        tmp_mat[1, 2] = dt
        tmp_mat[2, 2] = 1

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

    def __update(self, Y):
        self.IM = np.dot(self.H, self.X)
        self.IS = self.R + np.dot(self.H, np.dot(self.P, self.H.T))

        self.IN = Y - self.IM #innovation
        self.K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(self.IS))) #Gain
        self.X = self.X + np.dot(self.K, self.IN)
        self.P = self.P - np.dot(self.K, np.dot(self.IS, self.K.T))

        #self.cur_cov += self.IN[:, np.newaxis] @ self.IN[:, np.newaxis].T
        #self.Q = self.K @ np.divide(self.cur_cov , self.cur_step) @ self.K.T


    def filter_fast(self, seq, zero_fill=True):
        self.__init_matrices(seq[0])
        return filter_optimized(seq, self.A, self.B, self.X, self.P, self.U, self.Q, self.H, self.R, zero_fill)

    def filter_old(self, seq, zero_fill=True):
        #Needs to be in shape (Steps * Feats)
        self.__init_matrices(seq[0])

        seq_new = np.copy(seq)
        seq_new[0, :] = seq[0, :]

        for i in range(1, seq.shape[0]):
           seq_new[i, :] = self.__predict(True)
           if(~(zero_fill and np.all(seq[i] == 0))):
               self.__update(seq[i])
           #self.cur_step += 1

        return seq_new
    
    def filter_c(self, seq):

        size = ctypes.c_size_t
        dims = seq.shape

        if(self.init_mat_c):
            params = np.array([self.dt, self.P_mul, self.R_mul, self.Q_mul], dtype=np.float64)
            params = np.ascontiguousarray(params)
            res_mat = self.fun_(np.ascontiguousarray(seq) , size(dims[0]), size(dims[1]), params)
        else:
            #TODO: FIX
            self.__init_matrices(seq[0])
            matrices = self.get_matrices()
            pdbl_array = struct_c_kalman_mat()
            
            keys_struct = ["A", "B", "P", "Q", "H", "R", "X", "U"]

            for n in range(len(keys_struct)):
                setattr(pdbl_array, keys_struct[n], matrices[n].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

            res_mat = self.fun_(np.ascontiguousarray(seq) , size(dims[0]), size(dims[1]), pdbl_array)

        res_np = np.ctypeslib.as_array(res_mat, shape=(dims[0], dims[1]))
 
        #res_old = self.filter_old(seq) for comparison

        return res_np


    def filter(self, seq):
        seq = seq.astype(np.float64)
        seq_new = self.func_f(seq)
        return seq_new


@njit
def filter_optimized(seq, A, B, X, P, U, Q, H, R, zero_fill):
    
    seq_new = np.copy(seq)
    seq_new[0, :] = seq[0, :]

    for i in range(1, seq.shape[0]):

        X_temp = np.dot(A, X) + np.dot(B, U)
        P_temp = np.dot(A, np.dot(P, A.T)) + Q

        X = X_temp
        P = P_temp

        seq_new[i, :] =  X_temp[0::3]

        if(~(zero_fill and np.all(seq[i] == 0))):
            IM_ = np.dot(H, X)
            IS_ = R + np.dot(H, np.dot(P, H.T))
            IN_ = (seq[i] - IM_).astype(np.float64) #innovation
            K_ = np.dot(P, np.dot(H.T, np.linalg.inv(IS_))) #Gain
            X = X + np.dot(K_, IN_)
            P = P - np.dot(K_, np.dot(IS_, K_.T))

    return seq_new


def errcheck(result, func, args):
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())


