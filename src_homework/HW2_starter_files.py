
from pathlib import Path
from typing import List
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import os
import zipfile
from src_homework.config import COMMON_COLUMN


# data pre processing
base_data_folder_path = 'data/HMOG'
file_name_to_colume_names = {
    'Accelerometer.csv': ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
    'Activity.csv': ['ID', 'SubjectID', 'Start_time', 'End_time', 'Relative_Start_time', 'Relative_End_time',
                     'Gesture_scenario', 'TaskID', 'ContentID'],
    'Gyroscope.csv': ['Systime', 'EventTime', 'ActivityID', 'X', 'Y', 'Z', 'Phone_orientation'],
}
print('sa')
def unzip_file(parent_file):
    import zipfile, fnmatch, os

    pattern = '*.zip'
    for root, dirs, files in os.walk(parent_file):
        for filename in fnmatch.filter(files, pattern):
            print(os.path.join(root, filename))
            zipfile.ZipFile(os.path.join(root, filename)).extractall(os.path.join(root, os.path.splitext(filename)[0]))
            os.remove(os.path.join(root, filename))


def get_user_ids(path):
    """
    Get all user ids based on name of folders under "public_dataset/"
    :return: a list of user ids
    """

    file_name = os.listdir(path)
    user_id = [i for i in file_name if '.pdf' not in i and '.DS_Store' not in i]
    return user_id


def get_user_session_ids(user_id, parent_path):
    """
    Get all session ids for a specific user based on folder structure
    e.g. "public_dataset/100669/100669_session_13" has user_id=100669, session_id=13
    :param user_id: user id
    :return: list of user session ids
    """

    file_set = os.listdir(os.path.join(parent_path, user_id, user_id))
    session_id_set = [i.split('_')[-1] for i in file_set if i != '.DS_Store']

    return session_id_set

def read_file(parent_folder, user_id, user_session_id, file_name, colume_names):
    """
    Read one of the csv files for a user
    :param user_id: user id
    :param user_session_id: user session id
    :param file_name: csv file name (key of file_name_to_colume_names)
    :param colume_names: a list of column names of the csv file (value of file_name_to_colume_names)
    :return: content of the csv file as pandas DataFrame
    """
    import pandas as pd
    data = pd.read_csv(os.path.join(parent_folder, user_id, user_id,
                                    user_id + '_session_' + user_session_id,
                                    file_name), names = colume_names)

    return data


def get_user_session_data(parent_folder, user_id, user_session_id):
    """
    Combine accelerometer, gyroscope, and activity labels for a specific session of a user
    Note: Timestamps are ignored when joining accelerometer and gyroscope data.
    :param user_id: user id
    :param user_session_id: user session id
    :return: combined DataFrame for a session
    """
    colume_names_1 = ['Systime', 'EventTime', 'ActivityID', 'X_a', 'Y_a', 'Z_a', 'Phone_orientation']
    colume_names_2 = ['Systime', 'EventTime', 'ActivityID', 'X_g', 'Y_g', 'Z_g', 'Phone_orientation']

    sub_data1 = read_file(parent_folder, user_id, user_session_id, 'Accelerometer'+'.csv', colume_names_1)
    sub_data2 = read_file(parent_folder, user_id, user_session_id, 'Gyroscope'+'.csv', colume_names_2)


    sub_data1.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)
    sub_data2.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)

    data = sub_data1.merge(sub_data2, on = ['EventTime', 'ActivityID'], how = 'outer')

    return data

# pick the user as well as activities and extract 3 out of 6 features
user_ids = get_user_ids(base_data_folder_path)
user_id = user_ids[0]
user_session_id = '10'
data = get_user_session_data(base_data_folder_path, user_id, user_session_id)
data = data.sort_values(by = 'EventTime')
data = data.interpolate()
data = data.dropna()

import random
feature = random.sample(['X_a', 'Y_a', 'Z_a', 'X_g', 'Y_g', 'Z_g'], k=3)
data_feature = data[['EventTime', 'ActivityID'] + feature]
activity = list(set(data_feature['ActivityID']))


# visualize of the features you pick
from src_homework.utilis import time_series_plot
save_path = 'result_hw/event_time_series'
for fea  in feature:
    sub_data = data_feature[['EventTime', 'ActivityID', fea]].copy()
    time_series_plot(sub_data, fea, save_path)
    print(fea, 'finished')


def multiV_curvature(sub_data, t):
    """
    Calculate multi V curvature
    :param nbddata: neighborhood of time t_i containing (t, x(t), y(t), z(t)), 
    where x(t), y(t), z(t) are the 3 out of the 6 features. 
    :return: multi V curvature
    """
    from numpy.linalg import det, norm
    from numpy import cross, dot, polyfit, zeros
    import numpy as np

    nbddata = sub_data.copy()

    t0 = nbddata['EventTime'][0]
    nbddata['t'] = nbddata['EventTime'] - t0
    nbddata = nbddata.drop(['EventTime'], axis=1)

    not_common_col = [i for i in nbddata.columns if i not in COMMON_COLUMN]

    for col in not_common_col:
        v = polyfit(nbddata['t'], nbddata[col], 3)
        if col == not_common_col[0]:
            v_matrix = np.array(v)
        else:
            temp = np.array(v)
            v_matrix = np.vstack((v_matrix, temp))
    v_matrix = v_matrix[:, 1:]

    a1 = v_matrix[:, 0] + 2*v_matrix[:, 1]*t + 3*v_matrix[:, 2]*t**2
    a2 = 2*v_matrix[:, 1] + 6*v_matrix[:, 2]*t

    curvature = norm(cross(a1, a2), 2) / norm(a1,2)**3

    return curvature

def multiV_torsion(sub_data, t):
    """
    Calculate multi V torsion
    :param nbddata: neighborhood of time t_i containing (t, x(t), y(t), z(t)), 
    where x(t), y(t), z(t) are the 3 out of the 6 features. 
    :return: multi V torsion
    """
    from numpy.linalg import det, norm
    from numpy import cross, dot, polyfit, zeros
    import numpy as np

    nbddata = sub_data.copy()

    t0 = nbddata['EventTime'][0]
    nbddata['t'] = nbddata['EventTime'] - t0
    nbddata = nbddata.drop(['EventTime'], axis=1)

    not_common_col = [i for i in nbddata.columns if i not in COMMON_COLUMN]

    for col in not_common_col:
        v = polyfit(nbddata['t'], nbddata[col], 3)
        if col == not_common_col[0]:
            v_matrix = np.array(v)
        else:
            temp = np.array(v)
            v_matrix = np.vstack((v_matrix, temp))
    v_matrix = v_matrix[:, 1:]

    a1 = v_matrix[:, 0] + 2*v_matrix[:, 1]*t + 3*v_matrix[:, 2]*t**2
    a2 = 2*v_matrix[:, 1] + 6*v_matrix[:, 2]*t
    a3 = 6*v_matrix[:, 2]

    torsion = dot(cross(a1, a2), a3) / norm(cross(a1, a2), 2)**2

    return torsion

def get_neigbor_data(data, t, size):
    """
    get the neigbor data of t
    :param data:
    :param t:
    :param size:
    :return: nbdata
    """
    t_index = data.index[data['EventTime'] == t].tolist()[0]
    data_point_index = range(t_index - int(size / 2), t_index + int(size / 2)+1)
    nbddata = data.iloc[data_point_index, :]
    nbddata.reset_index(drop = True, inplace = True)

    return nbddata



# Calucate and plot curvature and torsion of the features you pick
# reduce the size of EventTime
try:
    os.makedirs((os.path.join('result_hw', 'curvature_torsion')))
except:
    pass

data_feature['EventTime'] = data_feature['EventTime']*10**(-10)

size = 201
start_point = int(size/2)
for act in activity:
    act_data = data_feature[data_feature['ActivityID'] == act].copy()
    end_point = len(act_data) - start_point
    act_data.drop(['ActivityID'], axis = 1, inplace = True)
    act_data = act_data.reset_index(drop = True)

    result_t = pd.DataFrame(columns = ['time', 'curvature', 'torsion'])
    for i in range(start_point, end_point):
        t = act_data['EventTime'][i]
        sub_data = get_neigbor_data(act_data, t, size)
        curvature = multiV_curvature(sub_data, t)
        torsion = multiV_torsion(sub_data, t)

        result_t.loc[-1] = [t, curvature, torsion]
        result_t.index = result_t.index + 1
        print(t, curvature, torsion)

    result_t.reset_index(drop=True, inplace=True)

    plt.subplot(211)
    plt.plot(result_t['time'], result_t['curvature'])
    plt.title('curvature')
    plt.subplot(212)
    plt.plot(result_t['time'], result_t['torsion'])
    plt.title('torsion')
    plt.savefig(os.path.join('result_hw', 'curvature_torsion', '{}.png'.format(act)))
    plt.show()
