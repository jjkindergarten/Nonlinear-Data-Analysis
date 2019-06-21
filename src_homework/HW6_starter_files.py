from pathlib import Path
from typing import List
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import os
import zipfile
from src_homework.config import COMMON_COLUMN
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np


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
    # colume_names_3 = ['Systime', 'EventTime', 'ActivityID', 'X_m', 'Y_m', 'Z_m', 'Phone_orientation']

    sub_data1 = read_file(parent_folder, user_id, user_session_id, 'Accelerometer'+'.csv', colume_names_1)
    sub_data2 = read_file(parent_folder, user_id, user_session_id, 'Gyroscope'+'.csv', colume_names_2)
    # sub_data3 = read_file(parent_folder, user_id, user_session_id, 'Magnetometer'+'.csv', colume_names_3)


    sub_data1.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)
    sub_data2.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)
    # sub_data3.drop(['Systime', 'Phone_orientation'], axis=1, inplace = True)

    data = sub_data1.merge(sub_data2, on = ['EventTime', 'ActivityID'], how = 'outer')
    # data = data.merge(sub_data3, on = ['EventTime', 'ActivityID'], how = 'outer')

    return data

# Define a normalize function
def normalize(v):
    """
    Calculate normalized vector
    :param v: input vector
    :return: normalized vector
    """
    from numpy.linalg import norm

    return v/norm(v)

def normalize_m(m):

    for i in range(m.shape[0]):
        m[i,:] = normalize(m[i,:])

    return m



# pick the user as well as activities and extract 3 out of 6 features
user_ids = get_user_ids(base_data_folder_path)
user_id = user_ids[0]
user_session_id = '10'
data = get_user_session_data(base_data_folder_path, user_id, user_session_id)
data = data.sort_values(by = 'EventTime')
data = data.set_index('EventTime')
data = data.interpolate(method = 'index')
data = data.dropna()
data = data.reset_index()

from itertools import combinations
feature_set = list(combinations(['X_a', 'Y_a', 'Z_a', 'X_g', 'Y_g', 'Z_g'], 3))



# visualize of the features you pick
from src_homework.utilis import time_series_plot
save_path1 = 'result_hw/hw6/event_time_series'
save_path2 = 'result_hw/hw6/sphere'

try:
    os.makedirs(save_path2)
except:
    pass
for i in range(len(feature_set)):

    feature = list(feature_set[i])
    data_feature = data[['EventTime', 'ActivityID'] + feature]
    activity = list(set(data_feature['ActivityID']))

    time_series_plot(data_feature, '_'.join(feature), save_path1)
    print(feature, 'finished')

    # Normalize df_pick3 by row
    variable_name = [i for i in data_feature.columns if i not in COMMON_COLUMN]
    plot_data = data_feature[variable_name].values
    norma_plot_data = normalize_m(plot_data)


    # Make data for the sphere surface
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    # Create a 3d plot
    ax = plt.figure().gca(projection='3d')

    # Plot the sphere surface
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect("equal")
    plt.tight_layout()

    # Plot the date points of normalized df_pick3 on the sphere
    ax.scatter(norma_plot_data[:,0], norma_plot_data[:,1], norma_plot_data[:,2], color="k",s=20)
    plt.savefig(os.path.join(save_path2, '_'.join(feature) + '.png'.format(str(i))))
    plt.show()
    plt.close()






