from os import walk
import tensorflow as tf
import pandas as pd
from scipy import stats

# pkg for reading the edf data
import mne


class DataLoader:

    def __init__(self,
                 path="/Users/robinhorn/Documents/UNI/Thesis/Datasets/sleep-edf-database-expanded-1.0.0/sleep-cassette/"):

        # path to database
        self.path = path

        # gather file names
        self.recording_files = []
        self.label_files = []
        for (dirpath, dirnames, filenames) in walk(self.path):
            for filename in filenames:
                if "PSG" in filename:
                    self.recording_files.append(filename)
                else:
                    self.label_files.append(filename)

        # eef feature names
        self.eeg_features = ["EEG Fpz-Cz", "EEG Pz-Oz"]

        # numbers for sleep stages
        self.stage_to_num = {"Sleep stage W": 0,
                             "Sleep stage 1": 1,
                             "Sleep stage 2": 2,
                             "Sleep stage 3": 3,
                             "Sleep stage 4": 4,
                             "Sleep stage R": 5,
                             "Sleep stage ?": 6,
                             "Movement time": 7}

        # epoch length in seconds
        self.epoch_length = 30

    def night_to_dataset(self, night_idx=0, batch_size=64):
        """
        param night_idx:
        param batch_size:
        return: tf dataset with shapes ((batch_size, channel_num*3000), (batch_size, 8))
        """

        x = mne.io.read_raw_edf(self.path + self.recording_files[night_idx], verbose=False).to_data_frame()
        x = x[["time", *self.eeg_features]]

        y = mne.read_annotations(self.path + self.label_files[night_idx]).to_data_frame()

        # convert time in ms to datatime for merging
        x["onset"] = pd.to_datetime(x['time'], unit='ms')

        # merge based on onset
        dataset = pd.merge_asof(x, y)
        dataset = dataset.drop('duration', axis=1)
        dataset = dataset.replace({"description": self.stage_to_num})

        # separate into features and labels
        input_features = dataset[self.eeg_features].to_numpy()
        input_features = input_features.reshape(-1, 100 * self.epoch_length * 2)

        labels = dataset["description"]
        labels = stats.mode(labels.values.reshape(-1, 100 * self.epoch_length), axis=1)[0]

        # create tf dataset
        input_ds = tf.data.Dataset.from_tensor_slices(input_features)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels).map(lambda label: tf.squeeze(tf.one_hot(label, 8)))

        dataset = tf.data.Dataset.zip((input_ds, labels_ds)).cache()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(64)

        return dataset

    def night_to_dataset_two_scales(self, window_length=200, stride=100):
        pass
