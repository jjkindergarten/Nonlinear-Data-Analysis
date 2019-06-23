from src_project.model.utilities import *
from numpy.linalg import norm, eig
from math import pi

def dynamic_model(q, T, w):
    """

    :param q: original orientation
    :param T: the length of time between two samples
    :param w: ?
    :return: return the next time step orientation vector
    """

    exp_w = exp_p(T/2*w)
    return circle_point(q, exp_w)

def quest_alg(acc_value, magn_value):
    """
    QUEST Alogrithm
    :param acc_value: a vector (3 * 1)
    :param magn_value: a vector (3 * 1)
    :return: the intial orientation (4 * 1)
    """
    g_n = np.array([[0], [0], [1]])
    m_n = np.array([[1], [0], [0]])

    g_b = acc_value/norm(acc_value,2)
    m_b = np.cross(g_b, np.cross(magn_value/norm(magn_value,2), g_b))

    A = -p_L(g_n) @ q_R(g_b) - p_L(m_n) @ q_R(m_b)

    w, v = eig(A)
    true_orientation = v[:,0]


    rau = 20/180*pi
    nosie_covar = np.diag([1,1,1]) * rau

    orientation_value = np.random.multivariate_normal(np.array([0, 0, 0]), nosie_covar)

    return circle_point(exp_p(orientation_value/2), true_orientation)

def ekf_update(q, P, T, y):
