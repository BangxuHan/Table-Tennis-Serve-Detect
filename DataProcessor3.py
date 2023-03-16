import pickle
import numpy as np
import pandas as pd

csv_pose_path = '/home/kls/data/tabletennisdata/migu_pose+scorev4.csv'
pkl_file_path = '/home/kls/data/tabletennisdata/migu_set(labelXscrwv4).pkl'


smooth_labels_step = 6
n_frames = 10
skip_frame = 0
main_idx_parts = [2, 3, 4, 5, 6, 7]  # 1.5
'''
    [[Nose_x Nose_y Nose_s]
     [LShoulder_x LShoulder_y LShoulder_s]
     [RShoulder_x RShoulder_y RShoulder_s]
     [LElbow_x LElbow_y LElbow_s]
     [RElbow_x RElbow_y RElbow_s]
     [LWrist_x LWrist_y LWrist_s]
     [RWrist_x RWrist_y RWrist_s]
     [LHip_x LHip_y LHip_s]
     [RHip_x RHip_y RHip_s]
     [LKnee_x LKnee_y LKnee_s]
     [RKnee_x RKnee_y RKnee_s]
     [LAnkle_x LAnkle_y LAnkle_s]
     [RAnkle_x RAnkle_y RAnkle_s]]
'''
# LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LElbow_x', 'LElbow_y', 'RElbow_x',
              'RElbow_y', 'LWrist_x', 'LWrist_y', 'RWrist_x', 'RWrist_y']

annot = pd.read_csv(csv_pose_path)

# Remove NaN
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0]
annot = annot.drop(idx)
# One-Hot Labels
label_onehot = pd.get_dummies(annot['label'])
annot = annot.drop('label', axis=1).join(label_onehot)
cols = label_onehot.columns.values


def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i + max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels


feature_set = np.empty((0, int(n_frames/(1+skip_frame)), 14, 3))
labels_set = np.empty((0, len(cols)))
vid_list = annot['video'].unique()
for vid in vid_list:
    print(f'Process on: {vid}')
    data = annot[annot['video'] == vid].reset_index(drop=True).drop(columns='video')

    # Label Smoothing
    esp = 0.1
    data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)
    data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)

    # Separate continuous frames
    frames = data['frame'].values
    frames_set = []
    fs = [0]
    for i in range(1, len(frames)):
        if frames[i] < frames[i - 1] + smooth_labels_step:
            fs.append(i)
        else:
            frames_set.append(fs)
            fs = [i]
    frames_set.append(fs)

    for fs in frames_set:
        xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 13, 3)
        # Scale pose normalize
        xys[:, :, :2] = scale_pose(xys[:, :, :2])
        # Add center point
        xys = np.concatenate((xys, np.expand_dims((xys[:, 1, :] + xys[:, 2, :]) / 2, 1)), axis=1)

        # Weighting main parts score
        scr = xys[:, :, -1].copy()
        scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
        # Mean score
        scr = scr.mean(1)

        # Targets
        lb = data.iloc[fs, -len(cols):].values
        # Apply points score mean to all labels
        lb = lb * scr[:, None]

        for i in range(xys.shape[0] - n_frames):
            feature_set = np.append(feature_set, xys[i:i+n_frames:1+skip_frame][None, ...], axis=0)
            labels_set = np.append(labels_set, lb[i:i+n_frames:1+skip_frame].mean(0)[None, ...], axis=0)

with open(pkl_file_path, 'wb') as f:
    pickle.dump((feature_set, labels_set), f)
    print('saved successful')
