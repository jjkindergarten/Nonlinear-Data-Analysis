import numpy as np
from math import pi
from numpy.linalg import norm


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
    result[1:] = p_0 * q_v + q_0 * p_v + np.cross(p_v, q_v)

    return result

def box_x(u):
    """
    speical math operation
    :param u: a vector (1, 3)
    :return: 3 by 3 matrix
    """

    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

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
    pL_22 = p[0]*np.diag([1,1,1]) + box_x(p[1:])

    pL_2 = np.hstack((pL_21, pL_22))

    pL = np.vstack((pL_1, pL_2))

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
    pL_22 = p[0]*np.diag([1,1,1])  - box_x(p[1:])

    pL_2 = np.hstack((pL_21, pL_22))

    pL = np.vstack((pL_1, pL_2))

    return pL

def quaternion_conjugate(q):
    q = np.atleast_2d(q)
    if q.shape[1]==3:
        q = unit_q(q)

    qConj = q * np.r_[1, -1,-1,-1]

    if q.shape[0]==1:
        qConj=qConj.ravel()

    return qConj

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

def exp_p_exact(eta):
    """
    exact way for quanternion exp
    :param eta:
    :return:
    """
    exp_eta = np.zeros(4)
    exp_eta[0] = np.cos(norm(eta))
    exp_eta[1:] = eta/norm(eta)*np.sin(norm(eta))

    return exp_eta

def transfer_q_to_R(quant):
    """
    trasfer the quaternion to rotation matrix
    :param q: the quanternion vector (4 * 1)
    :return: the 3 by 3 rotation matrix
    """
    q = unit_q(quant).T

    R = np.zeros((9, q.shape[1]))
    R[0] = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    R[1] = 2 * (q[1] * q[2] - q[0] * q[3])
    R[2] = 2 * (q[1] * q[3] + q[0] * q[2])
    R[3] = 2 * (q[1] * q[2] + q[0] * q[3])
    R[4] = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    R[5] = 2 * (q[2] * q[3] - q[0] * q[1])
    R[6] = 2 * (q[1] * q[3] - q[0] * q[2])
    R[7] = 2 * (q[2] * q[3] + q[0] * q[1])
    R[8] = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    if R.shape[1] == 1:
        return np.reshape(R, (3, 3))
    else:
        return R.T


def quaternion_representation(p):
    """

    :param p: a vector (3 * 1)
    :return: the quaternion representation of the vector
    """
    p_qua = np.zeros(4)
    p_qua[1:] = p

    return p_qua

def transfer_matirx_to_orientation(R):
    """
    transfer the rotation matrix to orientation (yaw, pitch and roll)
    :param R: the 3 by 3 Rotation matrix
    :return: (yaw, pitch, roll)
    """

    yaw = np.arctan(R[0,1]/R[0,0])*180/pi
    pitch = -np.arcsin(R[0,2])*180/pi
    roll = np.arctan(R[1,2]/R[2,2])*180/pi

    return yaw, pitch, roll

def transfer_quanternion_to_orientation(q):
    """
    transfer quanternion representation to orientation (yaw, pitch and roll)
    :param q: the quanternion representation
    :return: (yaw, pitch, roll)
    """

    R = transfer_q_to_R(q)
    yaw, pitch, roll = transfer_matirx_to_orientation(R)

    return [yaw, pitch, roll]


def q_inv(q):
    ''' Quaternion inversion
    Parameters
    ----------
    q: array_like, shape ([3,4],) or (N,[3/4])
        quaternion or quaternion vectors

    Returns
    -------
    qinv : inverse quaternion(s)

    Notes
    -----
    .. math::
          q^{-1} = \\frac{q_0 - \\vec{q}}{|q|^2}
    More info under
    http://en.wikipedia.org/wiki/Quaternion

    Examples
    --------
    >>>  quat.q_inv([0,0,0.1])
    array([-0., -0., -0.1])

    >>> quat.q_inv([[cos(0.1),0,0,sin(0.1)],
    >>> [cos(0.2),0,sin(0.2),0]])
    array([[ 0.99500417, -0.        , -0.        , -0.09983342],
           [ 0.98006658, -0.        , -0.19866933, -0.        ]])
    '''

    q = np.atleast_2d(q)
    if q.shape[1] == 3:
        return -q
    else:
        qLength = np.sum(q ** 2, 1)
        qConj = q * np.r_[1, -1, -1, -1]
        return (qConj.T / qLength).T


def normalize(v):
    ''' Normalization of a given vector (with image)

    Parameters
    ----------
    v : array (N,) or (M,N)
        input vector
    Returns
    -------
    v_normalized : array (N,) or (M,N)
        normalized input vector
    .. image:: ../docs/Images/vector_normalize.png
        :scale: 33%
    Example
    -------
    >>> skinematics.vector.normalize([3, 0, 0])
    array([[ 1.,  0.,  0.]])

    >>> v = [[np.pi, 2, 3], [2, 0, 0]]
    >>> skinematics.vector.normalize(v)
    array([[ 0.6569322 ,  0.41821602,  0.62732404],
       [ 1.        ,  0.        ,  0.        ]])

    Notes
    -----
    .. math::
        \\vec{n} = \\frac{\\vec{v}}{|\\vec{v}|}

    '''

    from numpy.linalg import norm

    if np.array(v).ndim == 1:
        vectorFlag = True
    else:
        vectorFlag = False

    v = np.double(np.atleast_2d(v))  # otherwise I get in trouble 2 lines down, if v is integer!
    length = norm(v, axis=1)
    v[length != 0] = (v[length != 0].T / length[length != 0]).T
    if vectorFlag:
        v = v.ravel()
    return v


def convert(rMat, to='quat'):
    """
    Converts a rotation matrix to the corresponding quaternion.
    Assumes that R has the shape (3,3), or the matrix elements in columns
    Parameters
    ----------
    rMat : array, shape (3,3) or (N,9)
        single rotation matrix, or matrix with rotation-matrix elements.
    to : string
        Currently, only 'quat' is supported

    Returns
    -------
    outQuat : array, shape (4,) or (N,4)
        corresponding quaternion vector(s)

    Notes
    -----
    .. math::
         \\vec q = 0.5*copysign\\left( {\\begin{array}{*{20}{c}}
        {\\sqrt {1 + {R_{xx}} - {R_{yy}} - {R_{zz}}} ,}\\\\
        {\\sqrt {1 - {R_{xx}} + {R_{yy}} - {R_{zz}}} ,}\\\\
        {\\sqrt {1 - {R_{xx}} - {R_{yy}} + {R_{zz}}} ,}
        \\end{array}\\begin{array}{*{20}{c}}
        {{R_{zy}} - {R_{yz}}}\\\\
        {{R_{xz}} - {R_{zx}}}\\\\
        {{R_{yx}} - {R_{xy}}}
        \\end{array}} \\right)

    More info under
    http://en.wikipedia.org/wiki/Quaternion

    Examples
    --------

    >>> rotMat = array([[cos(alpha), -sin(alpha), 0],
    >>>    [sin(alpha), cos(alpha), 0],
    >>>    [0, 0, 1]])
    >>> rotmat.convert(rotMat, 'quat')
    array([[ 0.99500417,  0.        ,  0.        ,  0.09983342]])

    """

    if to != 'quat':
        raise IOError('Only know "quat"!')

    if rMat.shape == (3, 3) or rMat.shape == (9,):
        rMat = np.atleast_2d(rMat.ravel()).T
    else:
        rMat = rMat.T
    q = np.zeros((4, rMat.shape[1]))

    R11 = rMat[0]
    R12 = rMat[1]
    R13 = rMat[2]
    R21 = rMat[3]
    R22 = rMat[4]
    R23 = rMat[5]
    R31 = rMat[6]
    R32 = rMat[7]
    R33 = rMat[8]

    # Catch small numerical inaccuracies, but produce an error for larger problems
    epsilon = 1e-10
    if np.min(np.vstack((1 + R11 - R22 - R33, 1 - R11 + R22 - R33, 1 - R11 - R22 + R33))) < -epsilon:
        raise ValueError('Problems with defintion of rotation matrices')

    q[1] = 0.5 * np.copysign(np.sqrt(np.abs(1 + R11 - R22 - R33)), R32 - R23)
    q[2] = 0.5 * np.copysign(np.sqrt(np.abs(1 - R11 + R22 - R33)), R13 - R31)
    q[3] = 0.5 * np.copysign(np.sqrt(np.abs(1 - R11 - R22 + R33)), R21 - R12)
    q[0] = np.sqrt(1 - (q[1] ** 2 + q[2] ** 2 + q[3] ** 2))

    return q.T


def unit_q(inData):
    ''' Utility function, which turns a quaternion vector into a unit quaternion.
    If the input is already a full quaternion, the output equals the input.
    Parameters
    ----------
    inData : array_like, shape (3,) or (N,3)
        quaternions or quaternion vectors

    Returns
    -------
    quats : array, shape (4,) or (N,4)
        corresponding unit quaternions.

    Notes
    -----
    More info under
    http://en.wikipedia.org/wiki/Quaternion

    Examples
    --------
    >>> quats = array([[0,0, sin(0.1)],[0, sin(0.2), 0]])
    >>> quat.unit_q(quats)
    array([[ 0.99500417,  0.        ,  0.        ,  0.09983342],
           [ 0.98006658,  0.        ,  0.19866933,  0.        ]])
    '''
    inData = np.atleast_2d(inData)
    (m, n) = inData.shape
    if (n != 3) & (n != 4):
        raise ValueError('Quaternion must have 3 or 4 columns')
    if n == 3:
        qLength = 1 - np.sum(inData ** 2, 1)
        numLimit = 1e-12
        # Check for numerical problems
        if np.min(qLength) < -numLimit:
            raise ValueError('Quaternion is too long!')
        else:
            # Correct for numerical problems
            qLength[qLength < 0] = 0
        outData = np.hstack((np.c_[np.sqrt(qLength)], inData))

    else:
        outData = inData

    return outData


def q_mult(p, q):
    '''
    Quaternion multiplication: Calculates the product of two quaternions r = p * q
    If one of both of the quaterions have only three columns,
    the scalar component is calculated such that the length
    of the quaternion is one.
    The lengths of the quaternions have to match, or one of
    the two quaternions has to have the length one.
    If both p and q only have 3 components, the returned quaternion
    also only has 3 components (i.e. the quaternion vector)

    Parameters
    ----------
    p,q : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors

    Returns
    -------
    r : quaternion or quaternion vector (if both
        p and q are contain quaternion vectors).

    Notes
    -----
    .. math::
        q \\circ p = \\sum\\limits_{i=0}^3 {q_i I_i} * \\sum\\limits_{j=0}^3 \\
        {p_j I_j} = (q_0 p_0 - \\vec{q} \\cdot \\vec{p}) + (q_0 \\vec{p} + p_0 \\
        \\vec{q} + \\vec{q} \\times \\vec{p}) \\cdot \\vec{I}

    More info under
    http://en.wikipedia.org/wiki/Quaternion

    Examples
    --------
    >>> p = [cos(0.2), 0, 0, sin(0.2)]
    >>> q = [[0, 0, 0.1],
    >>>    [0, 0.1, 0]]
    >>> r = quat.q_mult(p,q)
    '''

    flag3D = False
    p = np.atleast_2d(p)
    q = np.atleast_2d(q)
    if p.shape[1] == 3 & q.shape[1] == 3:
        flag3D = True

    if len(p) != len(q):
        assert (len(p) == 1 or len(q) == 1), \
            'Both arguments in the quaternion multiplication must have the same number of rows, unless one has only one row.'

    p = unit_q(p).T
    q = unit_q(q).T

    if np.prod(np.shape(p)) > np.prod(np.shape(q)):
        r = np.zeros(np.shape(p))
    else:
        r = np.zeros(np.shape(q))

    r[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
    r[1] = p[1] * q[0] + p[0] * q[1] + p[2] * q[3] - p[3] * q[2]
    r[2] = p[2] * q[0] + p[0] * q[2] + p[3] * q[1] - p[1] * q[3]
    r[3] = p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1]

    if flag3D:
        # for rotations > 180 deg
        r[:, r[0] < 0] = -r[:, r[0] < 0]
        r = r[1:]

    r = r.T
    return r


def rotate_vector(vector, q):
    '''
    Rotates a vector, according to the given quaternions.
    Note that a single vector can be rotated into many orientations;
    or a row of vectors can all be rotated by a single quaternion.


    Parameters
    ----------
    vector : array, shape (3,) or (N,3)
        vector(s) to be rotated.
    q : array_like, shape ([3,4],) or (N,[3,4])
        quaternions or quaternion vectors.

    Returns
    -------
    rotated : array, shape (3,) or (N,3)
        rotated vector(s)


    .. image:: ../docs/Images/vector_rotate_vector.png
        :scale: 33%

    Notes
    -----
    .. math::
        q \\circ \\left( {\\vec x \\cdot \\vec I} \\right) \\circ {q^{ - 1}} = \\left( {{\\bf{R}} \\cdot \\vec x} \\right) \\cdot \\vec I

    More info under
    http://en.wikipedia.org/wiki/Quaternion

    Examples
    --------
    >>> mymat = eye(3)
    >>> myVector = r_[1,0,0]
    >>> quats = array([[0,0, sin(0.1)],[0, sin(0.2), 0]])
    >>> quat.rotate_vector(myVector, quats)
    array([[ 0.98006658,  0.19866933,  0.        ],
           [ 0.92106099,  0.        , -0.38941834]])

    >>> quat.rotate_vector(mymat, [0, 0, sin(0.1)])
    array([[ 0.98006658,  0.19866933,  0.        ],
           [-0.19866933,  0.98006658,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    '''
    vector = np.atleast_2d(vector)
    qvector = np.hstack((np.zeros((vector.shape[0], 1)), vector))
    vRotated = q_mult(q, q_mult(qvector, q_inv(q)))
    vRotated = vRotated[:, 1:]

    if min(vRotated.shape) == 1:
        vRotated = vRotated.ravel()

    return vRotated