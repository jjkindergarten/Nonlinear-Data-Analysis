from src_project.model.utilities import *
import numpy as np

def Madgwick(q, Gyroscope, Accelerometer, Magnetometer):
    '''Calculate the best quaternion to the given measurement values.

    Parameters
    ----------
    q: quanternion vector
    Gyroscope : array, shape (N,3)
        Angular velocity [rad/s]
    Accelerometer : array, shape (N,3)
        Linear acceleration (Only the direction is used, so units don't matter.)
    Magnetometer : array, shape (N,3)
        Orientation of local magenetic field.
        (Again, only the direction is used, so units don't matter.)

    '''

    Beta = 1.0
    SamplePeriod = 0.01

    # Reference direction of Earth's magnetic field
    h = rotate_vector(Magnetometer, q)
    b = np.hstack((0, np.sqrt(h[0] ** 2 + h[1] ** 2), 0, h[2]))

    # Gradient decent algorithm corrective step
    F = [2 * (q[1] * q[3] - q[0] * q[2]) - Accelerometer[0],
         2 * (q[0] * q[1] + q[2] * q[3]) - Accelerometer[1],
         2 * (0.5 - q[1] ** 2 - q[2] ** 2) - Accelerometer[2],
         2 * b[1] * (0.5 - q[2] ** 2 - q[3] ** 2) + 2 * b[3] * (q[1] * q[3] - q[0] * q[2]) - Magnetometer[0],
         2 * b[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * (q[0] * q[1] + q[2] * q[3]) - Magnetometer[1],
         2 * b[1] * (q[0] * q[2] + q[1] * q[3]) + 2 * b[3] * (0.5 - q[1] ** 2 - q[2] ** 2) - Magnetometer[2]]

    J = np.array([
        [-2 * q[2], 2 * q[3], -2 * q[0], 2 * q[1]],
        [2 * q[1], 2 * q[0], 2 * q[3], 2 * q[2]],
        [0, -4 * q[1], -4 * q[2], 0],
        [-2 * b[3] * q[2], 2 * b[3] * q[3], -4 * b[1] * q[2] - 2 * b[3] * q[0], -4 * b[1] * q[3] + 2 * b[3] * q[1]],
        [-2 * b[1] * q[3] + 2 * b[3] * q[1], 2 * b[1] * q[2] + 2 * b[3] * q[0], 2 * b[1] * q[1] + 2 * b[3] * q[3],
         -2 * b[1] * q[0] + 2 * b[3] * q[2]],
        [2 * b[1] * q[2], 2 * b[1] * q[3] - 4 * b[3] * q[1], 2 * b[1] * q[0] - 4 * b[3] * q[2], 2 * b[1] * q[1]]])

    step = J.T.dot(F)
    step = normalize(step)  # normalise step magnitude

    # Compute rate of change of quaternion
    qDot = 0.5 * q_mult(q, np.hstack([0, Gyroscope])) - Beta * step

    # Integrate to yield quaternion
    q = q + qDot * SamplePeriod
    q = normalize(q).flatten()

    return q

if __name__ == "__main__":

    from src_project.data.data import *
    from src_project.data.visualization import *
    import matplotlib.pyplot as plt
    from math import pi

    main_path = 'data/HMOG'
    image_save_path = 'result_project/sensor_time_series'

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

    # 220962051000001 1 sitting and reading
    # 220962052000001 2 reading and walking
    # 220962053000001 3 writting and sitting
    # 220962054000001 4 writting and wlaking
    # 220962055000001 5 mapping and sitting
    # 220962056000001 6 mapping and walking

    test_data = data[data['ActivityID'] == 220962056000001]
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


    for i in range(len(test_gyroscope)):
        if i == 0:
            result = Madgwick(np.array([1, 0, 0, 0]), test_gyroscope[i,:], test_accelerometer[i,:], test_magnetometer[i,:])
            result = result.reshape([1,4])
        else:
            temp = Madgwick(result[(i-1),:], test_gyroscope[i,:], test_accelerometer[i,:], test_magnetometer[i,:])
            result = np.vstack((result, temp))

        print(i)

    ori_set = []
    for i in range(len(result)):
        ori_set.append(transfer_quanternion_to_orientation(result[i,:]))

    ori_data = pd.DataFrame(data = ori_set, columns= ['yaw', 'pitch', 'roll'])
    plt.plot(ori_data['yaw'][:100])
    plt.plot(ori_data['pitch'][:100])
    plt.plot(ori_data['roll'][:100])
    plt.show()
