import numpy as np

def circle_point(p, q):
    """
    special math operation
    :param p: a vector (1,4)
    :param q: a vector (1,4)
    :return: a vector with shape (1,4)
    """
    p_0 = p[0]
    p_v = p[1:]

    q_0 = q[0]
    q_v = q[1:]

    result = np.zeros([4])

    result[0] = p_0 * q_0 - p_v @ q_v
    result[1:3] = p_0 * q_v + q_0 * p_v + np.cross(p_v, q_v)

    return result

def box_x(u):
    """
    speical math operation
    :param u: a vector (1, 3)
    :return: 3 by 3 matrix
    """

    return np.array([[0, -u[2], u[1]], [u[3], 0, -u[1]], [-u[2], u[1], 0]])

def p_L(p):
    """
    special math operation
    :param p: a vector (1, 4)
    :return: a 3 by 3 matrix
    """

    pL_11 = p[0]
    pL_12 = -p[1:]

    pL_1 = np.hstack((pL_11, pL_12))

    pL_21 = p[1:].reshape([3,1])
    pL_22 = p[0]*np.ones([1,3]) + box_x(p[1:])

    pL_2 = np.hstack((pL_21, pL_22))

    pL = np.vstack(pL_1, pL_2)

    return pL

def q_R(p):
    """
    special math operation
    :param p: a vector (1, 4)
    :return: a 3 by 3 matrix
    """

    pL_11 = p[0]
    pL_12 = -p[1:]

    pL_1 = np.hstack((pL_11, pL_12))

    pL_21 = p[1:].reshape([3,1])
    pL_22 = p[0]*np.ones([1,3]) - box_x(p[1:])

    pL_2 = np.hstack((pL_21, pL_22))

    pL = np.vstack(pL_1, pL_2)

    return pL

def quaternion_conjugate(q):
    return np.hstack((q[0], -q[1:]))

def rotation(q, x):
    """
    rotation the current orientation given the rotation vector
    :param q: rotation matrix expressed as quaternion
    :param x: the orginal orientation (4 by 1) with the first entry to be 0
    :return: rotated vector (4 by 1) expressed as quaternion
    """

    result = circle_point(q, x)
    result = circle_point(result, quaternion_conjugate(q))

    return result

def rotation_quaternion(nv, a):
    """
    generate the rotation quaternion given the rotated angle and the unit vector
    :param nv: the unit vector which is orthgonal to the rotate angle
    :param a: the rotated angle
    :return: a vector (4 * 1) as rotation_quaternion
    """

    result1 = np.cos(a/2)
    result2 = -np.sin(a/2)*nv

    return np.hstack((result1, result2))

def exp_p(eta):

    return np.hstack((1, eta))

def transfer_q_to_R(q):
    """
    trasfer the quaternion to rotation matrix
    :param q: the quanternion vector (4 * 1)
    :return: the 3 by 3 rotation matrix
    """
    q_v = q[1:]
    q0 =  q[0]

    R1 = q_v.reshape([3,1]) @ q_v.reshape([1,3])
    R2 = q0**2 * np.diag([1,1,1])
    R3 = 2*q0*box_x(q_v) + box_x(q_v) @ box_x(q_v)

    return R1 + R2 + R3
