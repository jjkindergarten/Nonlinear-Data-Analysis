import numpy as np
from src_project.model.utilities import q_inv, normalize, convert, q_mult, transfer_quanternion_to_orientation

def kalman(rate, acc, omega, mag):
    '''
    Calclulate the orientation from IMU magnetometer data.

    Parameters
    ----------
    rate : float
    	   sample rate [Hz]
    acc : (N,3) ndarray
    	  linear acceleration [m/sec^2]
    omega : (N,3) ndarray
    	  angular velocity [rad/sec]
    mag : (N,3) ndarray
    	  magnetic field orientation

    Returns
    -------
    qOut : (N,4) ndarray
    	   unit quaternion, describing the orientation relativ to the coordinate system spanned by the local magnetic field, and gravity

    Notes
    -----
    Based on "Design, Implementation, and Experimental Results of a Quaternion-Based Kalman Filter for Human Body Motion Tracking" Yun, X. and Bachman, E.R., IEEE TRANSACTIONS ON ROBOTICS, VOL. 22, 1216-1227 (2006)

    '''

    numData = len(acc)

    # Set parameters for Kalman Filter
    tstep = 1. / rate
    tau = [0.5, 0.5, 0.5]  # from Yun, 2006

    # Initializations
    x_k = np.zeros(7)  # state vector
    z_k = np.zeros(7)  # measurement vector
    z_k_pre = np.zeros(7)
    P_k = np.matrix(np.eye(7))  # error covariance matrix P_k

    Phi_k = np.matrix(np.zeros((7, 7)))  # discrete state transition matrix Phi_k
    for ii in range(3):
        Phi_k[ii, ii] = np.exp(-tstep / tau[ii])

    H_k = np.eye(7)  # Identity matrix

    Q_k = np.zeros((7, 7))  # process noise matrix Q_k
    # D = 0.0001*np.r_[0.4, 0.4, 0.4]		# [rad^2/sec^2]; from Yun, 2006
    D = np.r_[0.4, 0.4, 0.4]  # [rad^2/sec^2]; from Yun, 2006

    for ii in range(3):
        Q_k[ii, ii] = D[ii] / (2 * tau[ii]) * (1 - np.exp(-2 * tstep / tau[ii]))

    # Evaluate measurement noise covariance matrix R_k
    R_k = np.zeros((7, 7))
    r_angvel = 0.01;  # [rad**2/sec**2]; from Yun, 2006
    r_quats = 0.0001;  # from Yun, 2006
    for ii in range(7):
        if ii < 3:
            R_k[ii, ii] = r_angvel
        else:
            R_k[ii, ii] = r_quats

    # Calculation of orientation for every time step
    qOut = np.zeros((numData, 4))

    for ii in range(numData):
        accelVec = acc[ii, :]
        magVec = mag[ii, :]
        angvelVec = omega[ii, :]
        z_k_pre = z_k.copy()  # watch out: by default, Python passes the reference!!

        # Evaluate quaternion based on acceleration and magnetic field data
        accelVec_n = normalize(accelVec)
        magVec_hor = magVec - accelVec_n * (accelVec_n @ magVec)
        magVec_n = normalize(magVec_hor)
        basisVectors = np.vstack((magVec_n, np.cross(accelVec_n, magVec_n), accelVec_n)).T
        quatRef = q_inv(convert(basisVectors, to='quat')).flatten()

        # Update measurement vector z_k
        z_k[:3] = angvelVec
        z_k[3:] = quatRef

        # Calculate Kalman Gain
        # K_k = P_k * H_k.T * inv(H_k*P_k*H_k.T + R_k)
        K_k = P_k @ np.linalg.inv(P_k + R_k)

        # Update state vector x_k
        x_k += np.array(K_k.dot(z_k - z_k_pre)).ravel()

        # Evaluate discrete state transition matrix Phi_k
        Phi_k[3, :] = np.r_[-x_k[4] * tstep / 2, -x_k[5] * tstep / 2, -x_k[6] * tstep / 2, 1, -x_k[0] * tstep / 2, -x_k[
            1] * tstep / 2, -x_k[2] * tstep / 2]
        Phi_k[4, :] = np.r_[
            x_k[3] * tstep / 2, -x_k[6] * tstep / 2, x_k[5] * tstep / 2, x_k[0] * tstep / 2, 1, x_k[2] * tstep / 2, -
            x_k[1] * tstep / 2]
        Phi_k[5, :] = np.r_[
            x_k[6] * tstep / 2, x_k[3] * tstep / 2, -x_k[4] * tstep / 2, x_k[1] * tstep / 2, -x_k[2] * tstep / 2, 1,
            x_k[0] * tstep / 2]
        Phi_k[6, :] = np.r_[
            -x_k[5] * tstep / 2, x_k[4] * tstep / 2, x_k[3] * tstep / 2, x_k[2] * tstep / 2, x_k[1] * tstep / 2, -x_k[
                0] * tstep / 2, 1]

        # Update error covariance matrix
        # P_k = (eye(7)-K_k*H_k)*P_k
        P_k = (np.eye(7) - K_k) @ P_k

        # Projection of state quaternions
        x_k[3:] += 0.5 * q_mult(x_k[3:], np.r_[0, x_k[:3]]).flatten()
        x_k[3:] = normalize(x_k[3:])
        x_k[:3] = np.zeros(3)
        x_k[:3] += tstep * (-x_k[:3] + z_k[:3])

        qOut[ii, :] = x_k[3:]

        # Projection of error covariance matrix
        P_k = Phi_k @ P_k @ Phi_k.T + Q_k

    # Make the first position the reference position
    qOut = q_mult(qOut, q_inv(qOut[0]))

    return qOut

if __name__ == "__main__":

    from src_project.data.data import *
    from src_project.data.visualization import *
    import matplotlib.pyplot as plt
    from math import pi

    main_path = 'data/HMOG'
    image_save_path = 'result_project/sensor_time_series'

    ids = get_user_ids(main_path)
    id = ids[0]
    id_session = '2'

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

    # 220962051000001 1 sitting and reading
    # 220962052000001 2 reading and walking
    # 220962053000001 3 writting and sitting
    # 220962054000001 4 writting and wlaking
    # 220962055000001 5 mapping and sitting
    # 220962056000001 6 mapping and walking

    test_data = data[data['ActivityID'] == 220962055000001]
    test_data.reset_index(drop=True, inplace=True)

    # initial uncertainty is set to be 20 degree
    rau = 20/180*pi
    initial_P = np.diag([1,1,1]) * rau**2
    sigma_acc = np.diag([1,1,1]) * 0.01**2
    sigma_gyr = np.diag([1,1,1]) * 0.001**2
    sigma_mag = np.diag([1,1,1]) * 0.01**2


    # step0
    test_accelerometer = test_data[['X_a', 'Y_a', 'Z_a']].values
    test_gyroscope = test_data[['X_g', 'Y_g', 'Z_g']].values
    test_magnetometer = test_data[['X_m', 'Y_m', 'Z_m']].values

    result3 = kalman(100, test_accelerometer, test_gyroscope, test_magnetometer)

    ori_set = []
    for i in range(len(result3)):
        ori_set.append(transfer_quanternion_to_orientation(result3[i,:]))

    ori_data = pd.DataFrame(data = ori_set, columns= ['yaw', 'pitch', 'roll'])
    plt.plot(ori_data['yaw'][:100])
    plt.plot(ori_data['pitch'][:100])
    plt.plot(ori_data['roll'][:100])
    plt.show()
