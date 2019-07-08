from src_project.model.main_model import quest_alg
from src_project.model.utilities import *
from numpy.linalg import norm, eig
from math import pi
from scipy.linalg import block_diag, inv

def ekf_update(q, P, T, y, sigma_y):
    """
    the update step in EKF
    :param q: the quaternian vector represents the orientation
    :param P: the covariance matrix for the quaternian matrix
    :param T: the time length between two samples
    :param y: the measurement value from Gyroscope sensors
    :return: the predicted next step oreintation  and its covariance matrix (and myabe the rotation matrix)
    """
    q_t_1 = circle_point(q, exp_p_exact(T/2 * y))

    der_exp_p_e = np.array([[0], [1], [1], [1]])

    F = q_R(exp_p_exact(T/2 * y))
    G = -T/2*p_L(q_t_1) @ der_exp_p_e

    P_t_1 = F @ P @ F.T + G @ sigma_y @ G.T

    return P_t_1, q_t_1

def ekf_correct(q_t_1, P_t_1, sigma_a, sigma_m, y_a, y_m):
    """
    the corrected step of EKF
    :param q_t_1: the updated orientation q_{t-1} from updated step
    :param P_t_1: the updated covariance matrix of orientation P_{t-1} from updated step
    :param sigma_a: the covariance matrix of accemetor sensor
    :param sigma_m: the covariance matrix of the magentic sensor
    :param y_a: the value gained from accemetor
    :param y_m: the value gained from magentic
    :return: orientation deviation eta_t,corrected orientation covariance matrix P_{t}
    """
    y_a = y_a.reshape([3,1])
    y_m = y_m.reshape([3,1])

    g_n = np.array([[0], [0], [1]])
    m_n = np.array([[1], [0], [0]])

    R_t_1 = transfer_q_to_R(q_t_1).T



