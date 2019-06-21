def time_series_plot(data, img_name, save_path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os
    import seaborn as sns
    from src_homework.config import COMMON_COLUMN

    # detect province
    activity_set = list(set(data.ActivityID))
    cols = 3
    num_subplot = len(activity_set)
    gs = gridspec.GridSpec(num_subplot // cols + 1, cols)
    figsize1 = ((num_subplot // cols + 1)*8, cols*6)
    fig = plt.figure(num_subplot, figsize=figsize1)

    ax = []
    try:
        os.makedirs(save_path)
    except:
        pass

    feature = [i for i in data.columns if i not in COMMON_COLUMN]
    col_act = [(feature, j) for j in activity_set]

    for i, case in enumerate(col_act):
        sub_data = data[['EventTime', 'ActivityID'] + case[0]]
        row = (i // cols)
        col = i % cols
        sub_data_prov = sub_data[sub_data['ActivityID'] == case[1]]
        sub_data_prov = sub_data_prov.drop(['ActivityID'], axis = 1)

        sub_data_prov.sort_values(by=['EventTime'], inplace=True)
        sub_data_prov.reset_index(inplace = True, drop = True)
        sub_data_prov_ts = sub_data_prov.set_index('EventTime')

        iv = sub_data_prov_ts.columns[0]

        ax.append(fig.add_subplot(gs[row, col]))

        ax[-1].set_title('{}_{}_time_series'.format(case[1], iv))
        ax[-1].plot(sub_data_prov_ts)
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, '{}.png'.format(img_name)))
    plt.show()
    plt.close()