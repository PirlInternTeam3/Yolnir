import glob
import numpy as np
import pandas as pd

path = './cnn/training_labeled_dataset/*.npz'
training_data = glob.glob(path)

height = 360
width = 640

frames = list()

# load data
for single_npz in training_data:
        with np.load(single_npz) as data:
            x = data['train']
            y = data['train_labels']
            y_df = pd.DataFrame(y, columns=['Forward', 'Right', 'Left'])
            frames.append(y_df)

result = pd.concat(frames)

total_num = len(result)
forward_num = len(result[result['Forward'] == 1])
right_num = len(result[result['Right'] == 1])
left_num = len(result[result['Left'] == 1])

print("\nTotal # of Labels: {}\nForward: {}, Right: {}, Left: {}".format(total_num, forward_num, right_num, left_num))