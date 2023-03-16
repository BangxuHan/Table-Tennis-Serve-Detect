import os

import cv2
import numpy as np
import pandas as pd

video_folder = '/home/kls/data/tabletennisdata/migu_new_data/All'
annot_file = '/home/kls/data/tabletennisdata/migu_new_data.csv'

# class_names = ['NotServe', 'Serve']
class_names = ['Rest', 'Serve', 'Hit']


def create_csv(folder, annot_file):
    label_num = 0  # 0:NotServe, 1:Serve
    list_file = sorted(os.listdir(folder))
    cols = ['video', 'frame', 'label']
    df = pd.DataFrame(columns=cols)
    for file in list_file:
        cap = cv2.VideoCapture(os.path.join(folder, file))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video = np.array([file] * frames_count)
        frame = np.arange(1, frames_count + 1)
        label = np.array([label_num] * frames_count)  # label serve ball videos 1; not serve videos 0
        rows = np.stack([video, frame, label], axis=1)
        df = df.append(pd.DataFrame(rows, columns=cols), ignore_index=True)
        cap.release()
    df.to_csv(annot_file, index=False)


if not os.path.exists(annot_file):
    create_csv(video_folder, annot_file)
    print('created successful')
else:
    print('file already exists')


index_video_to_play = 0  # Choose video to play
while True:
    annot = pd.read_csv(annot_file)
    video_list = annot.iloc[:, 0].unique()
    video_file = os.path.join(video_folder, video_list[index_video_to_play])
    print('Loading ({}) video {}'.format(index_video_to_play, video_file))
    # if index_video_to_play > len(video_list):
    #     break
    # else:
    #     index_video_to_play += 1
    # print('{}/{}'.format(index_video_to_play, len(video_list)), 'loading', os.path.basename(video_file))

    annot = annot[annot['video'] == video_list[index_video_to_play]].reset_index(drop=True)
    frames_idx = annot.iloc[:, 1].tolist()

    cap = cv2.VideoCapture(video_file)
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert frames_count == len(frames_idx), 'frame count not equal! {} and {}'.format(
        len(frames_idx), frames_count
    )

    i = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            cls_name = class_names[int(annot.iloc[i, -1])]

            if cls_name == 'Serve':
                color = (255, 0, 0)
            elif cls_name == 'Hit':
                color = (255, 128, 0)
            elif cls_name == 'Rest':
                color = (0, 255, 0)

            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame = cv2.putText(frame, 'Frame: {}/{} Pose: {}'.format(i + 1, frames_count, cls_name), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow('frame', frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                i += 1
                if i == frames_count:
                    break
                else:
                    continue
            elif key == ord('a'):
                i -= 1
                continue
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    index_video_to_play += 1
