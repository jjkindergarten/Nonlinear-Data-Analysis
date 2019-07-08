from src_project.model.utilities import *
from numpy.linalg import norm, eig
from math import pi
from scipy.linalg import block_diag, inv


def dynamic_model(q, T, w):
    """

    :param q: original orientation
    :param T: the length of time between two samples
    :param w: ?
    :return: return the next time step orientation vector
    """

    exp_w = exp_p_exact(T/2*w)
    return circle_point(q, exp_w)

def quest_alg(acc_value, magn_value):
    """
    QUEST Alogrithm
    :param acc_value: a vector (3 * 1)
    :param magn_value: a vector (3 * 1)
    :return: the intial orientation (4 * 1)
    """
    g_n = np.array([0, 0, 1])
    m_n = np.array([1, 0, 0])

    g_b = acc_value/norm(acc_value,2)
    m_b = np.cross(g_b, np.cross(magn_value/norm(magn_value,2), g_b))

    g_n_qua = quaternion_representation(g_n)
    m_n_qua = quaternion_representation(m_n)
    g_b_qua = quaternion_representation(g_b)
    m_b_qua = quaternion_representation(m_b)

    A = -p_L(g_n_qua) @ q_R(g_b_qua) - p_L(m_n_qua) @ q_R(m_b_qua)

    w, v = eig(A)
    true_orientation = v[:,0]


    rau = 20/180*pi
    noise_covar = np.diag([1,1,1]) * rau**2

    orientation_value = np.random.multivariate_normal(np.array([0, 0, 0]), noise_covar)

    return circle_point(exp_p_exact(orientation_value/2), true_orientation)

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
    R_t_1 = transfer_q_to_R(q_t_1)
    G_t_1 = T * R_t_1

    P_t_1 = P + G_t_1 @ sigma_y @ G_t_1.T

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
    H1 = -R_t_1 @ box_x(g_n)
    H2 = R_t_1 @ box_x(m_n)

    H = np.vstack((H1, H2))
    R = block_diag(sigma_a, sigma_m)

    S = H @ P_t_1 @ H.T + R
    k_t = P_t_1 @ H.T @ inv(S)

    y_t_1_acc = -R_t_1 @ g_n
    y_t_1_mag = R_t_1 @ m_n
    y_t_1 = np.vstack((y_t_1_acc, y_t_1_mag))
    y_t = np.vstack((y_a, y_m))
    epsilon = y_t - y_t_1

    eta_t = k_t @ epsilon
    P_t = P_t_1 - k_t @ S @ k_t.T

    return eta_t.reshape([3]), P_t

def relinearization(eta_t, q_t_1):
    """
    relinearize the orientation given by the Kalman filter
    :param eta_t: from corrected step
    :param q_t_1: from update step
    :return: q_t the orientation estimated from EKF
    """

    return circle_point(exp_p_exact(eta_t/2), q_t_1)

if __name__ == "__main__":

    from src_project.data.data import *
    from src_project.data.visualization import *
    import matplotlib.pyplot as plt
    from src_project.model.model3 import kalman
    from src_project.model.model4 import Madgwick
    from src_project.model.model5 import Mahony

    main_path = 'data/HMOG'
    image_save_path = 'result_project/sensor_time_series'
    result_save_Path = 'result_project/orientation_estimation'


    ids = get_user_ids(main_path)
    id = ids[0]
    id_session = '5'

    accelerometer, gyroscope, magnetometer = get_user_session_data(main_path, id, id_session)

    accelerometer.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)
    gyroscope.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)
    magnetometer.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)

    data = accelerometer.merge(gyroscope, on=['EventTime', 'ActivityID'], how='outer')
    data = data.merge(magnetometer, on=['EventTime', 'ActivityID'], how='outer')

    data = interpolate(data)

    time_series_plot(data[['X_a', 'Y_a', 'Z_a', 'EventTime', 'ActivityID']], 'accelerometer', image_save_path)
    time_series_plot(data[['X_g', 'Y_g', 'Z_g', 'EventTime', 'ActivityID']], 'gyroscope', image_save_path)
    time_series_plot(data[['X_m', 'Y_m', 'Z_m', 'EventTime', 'ActivityID']], 'magnetometer', image_save_path)

    # 220962051000001 1 reading and sitting
    # 220962052000001 2 reading and walking
    # 220962053000001 3 writting and sitting
    # 220962054000001 4 writting and wlaking
    # 220962055000001 5 mapping and sitting
    # 220962056000001 6 mapping and walking

    test_data = data[data['ActivityID'] == 220962054000001]
    test_data.reset_index(drop=True, inplace=True)

    activity = 'writting_walking'

    try:
        os.makedirs(os.path.join(result_save_Path, activity))
    except:
        pass

    # initial uncertainty is set to be 20 degree
    rau = 20/180*pi
    initial_P = np.diag([1,1,1]) * rau**2
    sigma_acc = np.diag([1,1,1]) * 0.01**2
    sigma_gyr = np.diag([1,1,1]) * 0.001**2
    sigma_mag = np.diag([1,1,1]) * 0.01**2

    # step0
    test_accelerometer = test_data[['X_a', 'Y_a', 'Z_a']][:500]
    test_gyroscope = test_data[['X_g', 'Y_g', 'Z_g']][:500]
    test_magnetometer = test_data[['X_m', 'Y_m', 'Z_m']][:500]

    ################### EKF self create ###################
    initial_est_orientation = quest_alg(test_accelerometer[['X_a', 'Y_a', 'Z_a']].values[0],
                                        test_magnetometer[['X_m', 'Y_m', 'Z_m']].values[0])

    est_orientation_set = []
    initial_orien = transfer_quanternion_to_orientation(initial_est_orientation)
    est_orientation_set.append(initial_orien)

    # test_est_orientation_set = []
    # test_est_orientation_set.append(initial_orien)
    # for step in range(1, test_data.shape[0]):
    #     est_orientation = quest_alg(test_accelerometer[['X_a', 'Y_a', 'Z_a']].values[step],
    #                                         test_magnetometer[['X_m', 'Y_m', 'Z_m']].values[step])
    #     test_est_orientation_set.append(transfer_quanternion_to_orientation(est_orientation))
    #     print(step, ' finished')
    #
    # test_est_orientation_vis = pd.DataFrame(data = test_est_orientation_set, columns=['yaw', 'pitch', 'roll'])

    #EKF
    q_t = initial_est_orientation
    P_t = initial_P
    for step in range(1, test_accelerometer.shape[0]):
        # T = test_data['time_gap'][step]
        T = 0.01
        y_gyr = test_gyroscope[['X_g', 'Y_g', 'Z_g']].values[step]
        y_acc = test_accelerometer[['X_a', 'Y_a', 'Z_a']].values[step]
        y_mag = test_magnetometer[['X_m', 'Y_m', 'Z_m']].values[step]

        P_t_1, q_t_1 = ekf_update(q_t, P_t, T, y_gyr, sigma_gyr)
        eta_t, P_t = ekf_correct(q_t_1, P_t_1, sigma_acc, sigma_mag, y_acc, y_mag)
        q_t = relinearization(eta_t, q_t_1)
        q_t = q_t/norm(q_t)

        temp_orien = transfer_quanternion_to_orientation(q_t)
        est_orientation_set.append(temp_orien)
        print(step, ' finished')

    est_orientation_vis_self_ekf = pd.DataFrame(data = est_orientation_set, columns=['yaw', 'pitch', 'roll'])

    test_accelerometer = test_accelerometer.values
    test_gyroscope = test_gyroscope.values
    test_magnetometer = test_magnetometer.values

    ############## ekf modeling func #############
    result3 = kalman(100, test_accelerometer, test_gyroscope, test_magnetometer)

    ori_set = []
    for i in range(len(result3)):
        ori_set.append(transfer_quanternion_to_orientation(result3[i,:]))

    ori_data_ekf = pd.DataFrame(data = ori_set, columns= ['yaw', 'pitch', 'roll'])


    ########### Madgwick ##############
    for i in range(len(test_gyroscope)):
        if i == 0:
            result2 = Madgwick(np.array([1, 0, 0, 0]), test_gyroscope[i,:], test_accelerometer[i,:], test_magnetometer[i,:])
            result2 = result2.reshape([1,4])
        else:
            temp = Madgwick(result2[(i-1),:], test_gyroscope[i,:], test_accelerometer[i,:], test_magnetometer[i,:])
            result2 = np.vstack((result2, temp))

        print(i)

    ori_set = []
    for i in range(len(result2)):
        ori_set.append(transfer_quanternion_to_orientation(result2[i,:]))

    ori_data_mad = pd.DataFrame(data = ori_set, columns= ['yaw', 'pitch', 'roll'])

    ############ Mahony #############
    for i in range(len(test_gyroscope)):
        if i == 0:
            result5 = Mahony(np.array([1, 0, 0, 0]), test_gyroscope[i,:], test_accelerometer[i,:], test_magnetometer[i,:], 1, 0)
            result5 = result5.reshape([1,4])
        else:
            temp = Mahony(result5[(i-1),:], test_gyroscope[i,:], test_accelerometer[i,:], test_magnetometer[i,:], 1, 0)
            result5 = np.vstack((result5, temp))

        print(i)

    ori_set = []
    for i in range(len(result5)):
        ori_set.append(transfer_quanternion_to_orientation(result5[i,:]))

    ori_data_mah = pd.DataFrame(data = ori_set, columns= ['yaw', 'pitch', 'roll'])


    ######## yaw estimation ##########

    plt.plot(est_orientation_vis_self_ekf['yaw'][:200], label = 'ekf1')
    plt.plot(ori_data_ekf['yaw'][:200], label = 'ekf2')
    plt.plot(ori_data_mad['yaw'][:200], label = 'Madgwick')
    plt.plot(ori_data_mah['yaw'][:200], label= 'Mahony')
    plt.legend()
    plt.savefig(os.path.join(result_save_Path, activity, 'yaw.png'))
    plt.show()
    plt.close()

    ######## pitch estimation ##########

    plt.plot(est_orientation_vis_self_ekf['pitch'][:200], label = 'ekf1')
    plt.plot(ori_data_ekf['pitch'][:200], label = 'ekf2')
    plt.plot(ori_data_mad['pitch'][:200], label = 'Madgwick')
    plt.plot(ori_data_mah['pitch'][:200], label= 'Mahony')
    plt.legend()
    plt.savefig(os.path.join(result_save_Path, activity, 'pitch.png'))
    plt.show()
    plt.close()

    ######## roll estimation ##########

    plt.plot(est_orientation_vis_self_ekf['roll'][:200], label = 'ekf1')
    plt.plot(ori_data_ekf['roll'][:200], label = 'ekf2')
    plt.plot(ori_data_mad['roll'][:200], label = 'Madgwick')
    plt.plot(ori_data_mah['roll'][:200], label= 'Mahony')
    plt.legend()
    plt.savefig(os.path.join(result_save_Path, activity, 'roll.png'))
    plt.show()
    plt.close()

