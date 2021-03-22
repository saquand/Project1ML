import os
import random

import cv2 as cv
import typing
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf


def get_img_path_df(dir_: str, col_prefix: str) -> pd.DataFrame:
    """
    Create a pandas dataframe with the paths to valid '.nii.gz' files in it

    :param dir_:
    :param col_prefix: col prefix to assign name '{col_prefix}_img_path' to the column with image paths
    :return:
    """
    img_path_list = [os.path.join(dir_, filename) for filename in os.listdir(dir_) if
                     filename != '.DS_Store' and '._' not in filename]
    img_num_list = [filename.split('colon_')[1].split('.nii.gz')[0] for filename in os.listdir(dir_) if
                    filename != '.DS_Store' and '._' not in filename]

    img_path_df = pd.DataFrame({f'{col_prefix}_img_path': img_path_list,
                                'index': img_num_list}) \
        .set_index('index')

    return img_path_df


def add_depth_image(df_: pd.DataFrame, col_name: str) -> List:
    """
    Return a list with the depth of each of the image paths listed in `col_name`

    :param df_:
    :param col_name:
    :return:
    """

    channel_number_list = list()

    for index, img_path in df_[col_name].iteritems():
        channel_number_list.append(nib.load(img_path).shape[-1])

    return channel_number_list


def create_depth_based_index(df_: pd.DataFrame, col_to_use: str = 'depth') -> pd.DataFrame:
    """
    Create a new set of rows and indexes where we have a row for each possible image and depth/channel/cut it has
    :param df_:
    :param col_to_use:
    :return:
    """
    df_ = pd.DataFrame(df_[col_to_use].map(lambda depth: list(range(depth))).explode().rename('depth_i'))\
            .join(df_) \
            .set_index('depth_i', append=True)

    return df_


def build_train_test_df(data_path_source_dir_: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use the functions above to create a dataset for the train and test sets where we have as indexes the number of the
    image, each channel or depth it has, and the path to it.

    For train we generate a column for the path of the image and to the labels
    For train we only generate a column for the path of the images

    :param data_path_source_dir_:
    :return:
    """

    x_dir_path_tr = os.path.join(data_path_source_dir_, 'imagesTr')
    y_dir_path_tr = os.path.join(data_path_source_dir_, 'labelsTr')

    x_tr_df = get_img_path_df(dir_=x_dir_path_tr, col_prefix='x_tr')
    x_tr_df['x_tr_img_depth'] = add_depth_image(df_=x_tr_df, col_name='x_tr_img_path')

    y_tr_df = get_img_path_df(dir_=y_dir_path_tr, col_prefix='y_tr')
    y_tr_df['y_tr_img_depth'] = add_depth_image(df_=y_tr_df, col_name='y_tr_img_path')

    tr_df_ = x_tr_df.join(y_tr_df, how='inner')

    assert (tr_df_.x_tr_img_depth == tr_df_.y_tr_img_depth).all()

    tr_df_ = tr_df_.drop('y_tr_img_depth', axis=1).rename(columns={'x_tr_img_depth': 'depth'})

    tr_df_ = create_depth_based_index(df_=tr_df_, col_to_use='depth')

    # Convert to series
    x_dir_path_ts = os.path.join(data_path_source_dir_, 'imagesTs')
    x_ts_df_ = get_img_path_df(dir_=x_dir_path_ts, col_prefix='x_ts')
    x_ts_df_['depth'] = add_depth_image(df_=x_ts_df_, col_name='x_ts_img_path')
    x_ts_df_ = create_depth_based_index(df_=x_ts_df_, col_to_use='depth')

    return tr_df_, x_ts_df_


class DataGenerator2D(tf.keras.utils.Sequence):

    def __init__(self, df: pd.DataFrame, x_col: str, y_col: Optional[str] = None, batch_size: int = 32,
                 num_classes: Optional[int] = None, shuffle: bool = False, shuffle_depths: bool = False,
                 class_sampling: Optional[dict] = None, depth_class_col: Optional[str] = None,
                 resize_dim: Optional[tuple] = None, hounsfield_min: float = -1000., hounsfield_max: float = 2000.,
                 rotate_range: Optional[float] = None, horizontal_flip: bool = False, vertical_flip: bool = False,
                 random_crop: Optional[tuple] = None, shearing: Optional[Tuple[tuple, tuple]] =None,
                 gaussian_blur: Optional[Tuple[float, float]] = None
                 ):
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df[[]]
        self.num_classes = num_classes
        self.x_col = x_col
        self.y_col = y_col

        # Row shuffling, either images or images and their depths
        self.shuffle = shuffle
        self.shuffle_depths = shuffle_depths

        # Preprocessing attributes
        self.resize_dim = resize_dim
        self.hounsfield_min = hounsfield_min
        self.hounsfield_max = hounsfield_max

        # Augmentation operations and their parameters
        self.aug_operations = self.get_aug_operations(
            rotate_range=rotate_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            random_crop=random_crop,
            shearing=shearing,
            gaussian_blur=gaussian_blur
        )

        self.class_sampling = class_sampling
        self.depth_class_col = depth_class_col

        if self.class_sampling is not None and self.depth_class_col is not None:
            self.has_cancer_idx = self.df[self.df.has_cancer_pixels][[]]
            self.not_has_cancer_idx = self.df[~self.df.has_cancer_pixels][[]]

        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        """
        Here index is the index of the mini-batch, so it will be from 0 to the number of steps per epoch. What we do
        then is that we grab batch_size rows from the shuffled or unshuffled dataframe indexes.

        Remember that when we shuffle the indexes, we do it using it only an image's number, meaning the order of the
        images cut's is actually preserved

        :param index: mini-batch index number within an epoch. So if index=1, it means we are in mini-batch 2.
        :return:
        """

        # We use iloc to identify the next `batch_size` rows to pick, and then get their actual indexes (index, depth_i)
        #  to get get the corresponding path and images with the method __get_data()
        batch_idx = self.indices.iloc[pd.IndexSlice[index * self.batch_size: (index + 1) * self.batch_size, :]].index

        X, y = self.__get_data(batch_idx)

        return X, y

    def on_epoch_end(self):
        """
        After each epoch ends we usually shuffle the training data. This specific function shuffles the `self.indices`
        attribute indices either at image number, and image number and their cuts/depth if so chosen with the
        attributes `self.shuffle` and `self.shuffle_depths` at initialization

        :return:
        """

        # Implement up and down sampling of rows that do not contain cancer, the rows that make it to the sample changes
        #   each epoch
        if self.class_sampling is not None and self.depth_class_col is not None:
            epoch_cancer_idx = self.has_cancer_idx.groupby(level=0).sample(
                frac=self.class_sampling['cancer_pixel'], replace=True)
            epoch_not_cancer_idx = self.not_has_cancer_idx.groupby(level=0).sample(
                frac=self.class_sampling['not_cancer_pixel'])

            epoch_idx = pd.concat([epoch_cancer_idx, epoch_not_cancer_idx], axis=0)
            self.indices = epoch_idx.sort_index()

        if self.shuffle:
            # Shuffle indexes
            shuffled_img_idx = np.random.choice(self.indices.index.levels[0], self.indices.index.levels[0].shape[0],
                                                replace=False)

            if self.shuffle_depths:
                # Reshufle depths of each image. This could help as cuts that have cancer tend to occur consecutively
                self.indices = self.indices.loc[pd.IndexSlice[shuffled_img_idx, :], :]\
                                .groupby(level=0, sort=False).sample(frac=1)
            else:
                # Reorder index dataframe with new image indices but maintaining depth order
                self.indices = self.indices.loc[pd.IndexSlice[shuffled_img_idx, :], :]

    def __get_data(self, batch_idx: pd.MultiIndex) -> Tuple[np.ndarray, np.ndarray]:
        y = None
        current_image_path = ''
        x_img = None
        y_img = None
        y_i = None

        for i, (index, row) in enumerate(self.df.loc[batch_idx].iterrows()):

            # Only load a new CT scan if we need to, as we are iterating over the cuts
            if row[self.x_col] != current_image_path:
                x_img = nib.load(row[self.x_col]).get_fdata()

                if self.y_col is not None:
                    y_img = nib.load(row[self.y_col]).get_fdata()

                current_image_path = row[self.x_col]

            # Extract the channel/depth indicated in the second level of the index (depth_i)
            x_i = x_img[:, :, index[1]]

            if self.y_col is not None:
                y_i = y_img[:, :, index[1]]

            # Preprocess image
            x_i = self.preprocess_img(x_i)

            if self.y_col is not None:
                y_i = self.preprocess_img(y_i, label=True)

            # Apply augmentation operations to both x_i and y_i (if available)
            # Choose randomly one of the available operations to augment
            if len(self.aug_operations) > 1:
                # Choose randomly one of the augmentation transformations and generate it's parameters, if needed
                #   We need to generate them randomly before hand because they need to be the same for the image
                #   and the label mask
                aug_operation_i = random.choice(list(self.aug_operations.keys()))
                operation_param_dict = self.generate_augment_img_params(
                    img_shape=x_i.shape, aug_operation_=aug_operation_i)

                x_i = self.augment_img(img_=x_i, aug_operation_=aug_operation_i,
                                       operation_param_dict_=operation_param_dict)

                if self.y_col is not None:
                    y_i = self.augment_img(img_=y_i, aug_operation_=aug_operation_i,
                                           operation_param_dict_=operation_param_dict)

            # Reshape before adding to the mini-batch
            x_i = x_i.reshape((1,) + x_i.shape)
            if self.y_col is not None:
                y_i = y_i.reshape((1,) + y_i.shape)

            # Add them to the mini-batch
            if i == 0:
                X = x_i
                if self.y_col is not None:
                    y = y_i
            else:
                X = np.concatenate([X, x_i], axis=0)
                if self.y_col is not None:
                    y = np.concatenate([y, y_i], axis=0)

        return X, y

    @staticmethod
    def get_aug_operations(**kwargs) -> Dict:
        total_aug_operation_list = ['rotate_range', 'horizontal_flip', 'vertical_flip', 'random_crop', 'shearing',
                                    'gaussian_blur']
        aug_operations = {key: value for key, value in kwargs.items() if value is not None and value
                          and key in total_aug_operation_list}

        aug_operations['identity'] = True

        return aug_operations

    def preprocess_img(self, img_: np.ndarray, label: bool = False) -> np.ndarray:

        if not label:
            # Cap the pixel values
            img_[img_ < self.hounsfield_min] = self.hounsfield_min
            img_[img_ > self.hounsfield_max] = self.hounsfield_max

            # Normalize the image's intensity range
            hounsfield_range = self.hounsfield_max - self.hounsfield_min
            img_ = (img_ - self.hounsfield_min) / hounsfield_range

        # Return them normalized
        if self.resize_dim is not None:
            img_ = cv.resize(img_, self.resize_dim, interpolation=cv.INTER_AREA)

        return img_

    def generate_augment_img_params(self, img_shape: tuple, aug_operation_: str) -> Optional[dict]:
        # Generate the random parameters, if needed, for a transformation so that we can use the same both from the
        #  image and the label's mask

        param_dict_ = dict()
        if aug_operation_ == 'rotate_range':
            # Generate the random angle to use to rotate the image
            angle = np.random.uniform(-self.aug_operations[aug_operation_], self.aug_operations[aug_operation_])
            param_dict_['angle'] = angle
            return param_dict_

        elif aug_operation_ == 'random_crop':
            # Generate starting and ending positions for the random crop, both for the height and the width
            min_range = self.aug_operations[aug_operation_][0]
            max_range = self.aug_operations[aug_operation_][1]

            assert max_range < 1 or min_range > 0

            zoom_amount = random.uniform(min_range, max_range)
            h, w = img_shape[:2]
            h_taken = int(zoom_amount * h)
            w_taken = int(zoom_amount * w)
            h_start = random.randint(0, h - h_taken)
            w_start = random.randint(0, w - w_taken)
            h_end = h_start + h_taken
            w_end = w_start + w_taken
            param_dict_.update({'h_start': h_start, 'w_start': w_start, 'h_end': h_end, 'w_end': w_end})

            return param_dict_

        elif aug_operation_ == 'shearing':
            x_range = self.aug_operations[aug_operation_][0]
            y_range = self.aug_operations[aug_operation_][1]

            x_shear = random.uniform(x_range[0], x_range[1])
            y_shear = random.uniform(y_range[0], y_range[1])

            param_dict_.update({'x_shear': x_shear, 'y_shear': y_shear})

            return param_dict_

        elif aug_operation_ == 'gaussian_blur':
            sigma = random.uniform(self.aug_operations[aug_operation_][0], self.aug_operations[aug_operation_][1])
            param_dict_['sigma'] = sigma

            return param_dict_

        else:
            # This means there are no random parameters to generate for an augmentation transformation
            return None

    @staticmethod
    def augment_img(img_: np.ndarray, aug_operation_: str, operation_param_dict_: Optional[dict] = None) -> np.ndarray:

        if aug_operation_ == 'rotate_range':
            angle = operation_param_dict_['angle']
            (h, w) = img_.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
            rotated_img_ = cv.warpAffine(img_, M, (w, h))

            return rotated_img_

        elif aug_operation_ == 'horizontal_flip':
            img_ = cv.flip(img_, 1)
            return img_

        elif aug_operation_ == 'vertical_flip':
            img_ = cv.flip(img_, 0)
            return img_

        elif aug_operation_ == 'random_crop':
            h, w = img_.shape[0], img_.shape[1]
            img_ = img_[operation_param_dict_['h_start']: operation_param_dict_['h_end'],
                        operation_param_dict_['w_start']: operation_param_dict_['w_end']]
            img_ = cv.resize(img_, (h, w), interpolation=cv.INTER_AREA)
            return img_

        elif aug_operation_ == 'shearing':
            h, w = img_.shape[0], img_.shape[1]
            M2 = np.float32([[1, operation_param_dict_['x_shear'], 0],
                             [operation_param_dict_['y_shear'], 1, 0]])
            M2[0, 2] = -M2[0, 1] * h / 2
            M2[1, 2] = -M2[1, 0] * w / 2
            img_ = cv.warpAffine(img_, M2, (w, h))

            return img_

        elif aug_operation_ == 'gaussian_blur':
            filter_size = (0, 0)  # Size will be calculated based on the value of sigmaX
            img_ = cv.GaussianBlur(img_, filter_size,
                                   sigmaX=operation_param_dict_['sigma'], borderType=cv.BORDER_DEFAULT)
            return img_

        elif aug_operation_ == 'identity':
            return img_

        else:
            raise Exception('Augmented operation not programmed')


class DataGenerator3D(tf.keras.utils.Sequence):

    def __init__(self, df: pd.DataFrame, x_col: str, y_col: Optional[str] = None, batch_size: int = 2,
                 num_classes: Optional[int] = None, shuffle: bool = False,
                 resize_dim: Optional[tuple] = None, hounsfield_min: float = -1000., hounsfield_max: float = 400.,
                 rotate_range: Optional[float] = None, horizontal_flip: bool = False, vertical_flip: bool = False,
                 random_crop: Optional[tuple] = None, shearing: Optional[Tuple[tuple, tuple]] = None,
                 gaussian_blur: Optional[Tuple[float, float]] = None
                 ):
        self.batch_size = batch_size
        self.df = df
        self.indices = self.df[[]]
        self.num_classes = num_classes
        self.x_col = x_col
        self.y_col = y_col

        # Row shuffling, either images or images and their depths
        self.shuffle = shuffle

        # Preprocessing attributes
        self.resize_dim = resize_dim
        self.hounsfield_min = hounsfield_min
        self.hounsfield_max = hounsfield_max

        # Augmentation operations and their parameters
        self.aug_operations = self.get_aug_operations(
            rotate_range=rotate_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            random_crop=random_crop,
            shearing=shearing,
            gaussian_blur=gaussian_blur
        )

        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        """
        Here index is the index of the mini-batch, so it will be from 0 to the number of steps per epoch. What we do
        then is that we grab batch_size rows from the shuffled or unshuffled dataframe indexes.

        Remember that when we shuffle the indexes, we do it using it only an image's number, meaning the order of the
        images cut's is actually preserved

        :param index: mini-batch index number within an epoch. So if index=1, it means we are in mini-batch 2.
        :return:
        """

        # We use iloc to identify the next `batch_size` rows to pick and then read these images in 3D
        #  to get get the corresponding path and images with the method __get_data()
        batch_idx = self.indices.iloc[index * self.batch_size: (index + 1) * self.batch_size].index

        X, y = self.__get_data(batch_idx)

        # If the depth is odd, pad it with zeroes to make it even
        if X.shape[3] % 2 == 1:
            X = self.pad_along_axis(array=X, target_length=X.shape[3] + 1, axis=3)
            y = self.pad_along_axis(array=y, target_length=y.shape[3] + 1, axis=3)

        X = X.reshape(X.shape + (1,))
        y = X.reshape(y.shape + (1,))

        return X, y

    def on_epoch_end(self):
        """
        After each epoch ends we usually shuffle the training data. This specific function shuffles the `self.indices`
        attribute indices either at image number, and image number and their cuts/depth if so chosen with the
        attributes `self.shuffle` and `self.shuffle_depths` at initialization

        :return:
        """

        if self.shuffle:
            # Shuffle indexes
            shuffled_img_idx = np.random.choice(self.indices.index, self.indices.index.shape[0], replace=False)

            self.indices = self.indices.loc[shuffled_img_idx]

    def __get_data(self, batch_idx: pd.MultiIndex) -> Tuple[np.ndarray, np.ndarray]:
        y = None

        for i, (index, row) in enumerate(self.df.loc[batch_idx].iterrows()):

            # Read the 3D image
            x_img = nib.load(row[self.x_col]).get_fdata()

            if self.y_col is not None:
                y_img = nib.load(row[self.y_col]).get_fdata()

            # Preprocess image
            x_img = self.preprocess_img(x_img)

            if self.y_col is not None:
                y_img = self.preprocess_img(y_img, label=True)

            # Apply augmentation operations to both x_i and y_i (if available)
            # Choose randomly one of the available operations to augment
            if len(self.aug_operations) > 1:
                # Choose randomly one of the augmentation transformations and generate it's parameters, if needed
                #   We need to generate them randomly before hand because they need to be the same for the image
                #   and the label mask
                aug_operation_i = random.choice(list(self.aug_operations.keys()))
                operation_param_dict = self.generate_augment_img_params(
                    img_shape=x_img.shape, aug_operation_=aug_operation_i)

                x_img = self.augment_img(img_=x_img, aug_operation_=aug_operation_i,
                                         operation_param_dict_=operation_param_dict)

                if self.y_col is not None:
                    y_img = self.augment_img(img_=y_img, aug_operation_=aug_operation_i,
                                             operation_param_dict_=operation_param_dict)

            # If the image has an odd depth, pad it with zeroes along the last dimesion

            # Reshape before adding to the mini-batch
            x_img = x_img.reshape((1,) + x_img.shape)
            if self.y_col is not None:
                y_img = y_img.reshape((1,) + y_img.shape)

            # Add them to the mini-batch
            if i == 0:
                X = x_img
                if self.y_col is not None:
                    y = y_img
            else:
                X = np.concatenate([X, x_img], axis=0)
                if self.y_col is not None:
                    y = np.concatenate([y, y_img], axis=0)

        return X, y

    @staticmethod
    def get_aug_operations(**kwargs) -> Dict:
        total_aug_operation_list = ['rotate_range', 'horizontal_flip', 'vertical_flip', 'random_crop', 'shearing',
                                    'gaussian_blur']
        aug_operations = {key: value for key, value in kwargs.items() if value is not None and value
                          and key in total_aug_operation_list}

        aug_operations['identity'] = True

        return aug_operations

    def preprocess_img(self, img_: np.ndarray, label: bool = False) -> np.ndarray:

        if not label:
            # Cap the pixel values
            img_[img_ < self.hounsfield_min] = self.hounsfield_min
            img_[img_ > self.hounsfield_max] = self.hounsfield_max

            # Normalize the image's intensity range
            hounsfield_range = self.hounsfield_max - self.hounsfield_min
            img_ = (img_ - self.hounsfield_min) / hounsfield_range

        # Return them normalized
        if self.resize_dim is not None:
            img_ = cv.resize(img_, self.resize_dim, interpolation=cv.INTER_AREA)

        return img_

    def generate_augment_img_params(self, img_shape: tuple, aug_operation_: str) -> Optional[dict]:
        # Generate the random parameters, if needed, for a transformation so that we can use the same both from the
        #  image and the label's mask

        param_dict_ = dict()
        if aug_operation_ == 'rotate_range':
            # Generate the random angle to use to rotate the image
            angle = np.random.uniform(-self.aug_operations[aug_operation_], self.aug_operations[aug_operation_])
            param_dict_['angle'] = angle
            return param_dict_

        elif aug_operation_ == 'random_crop':
            # Generate starting and ending positions for the random crop, both for the height and the width
            min_range = self.aug_operations[aug_operation_][0]
            max_range = self.aug_operations[aug_operation_][1]

            assert max_range < 1 or min_range > 0

            zoom_amount = random.uniform(min_range, max_range)
            h, w = img_shape[:2]
            h_taken = int(zoom_amount * h)
            w_taken = int(zoom_amount * w)
            h_start = random.randint(0, h - h_taken)
            w_start = random.randint(0, w - w_taken)
            h_end = h_start + h_taken
            w_end = w_start + w_taken
            param_dict_.update({'h_start': h_start, 'w_start': w_start, 'h_end': h_end, 'w_end': w_end})

            return param_dict_

        elif aug_operation_ == 'shearing':
            x_range = self.aug_operations[aug_operation_][0]
            y_range = self.aug_operations[aug_operation_][1]

            x_shear = random.uniform(x_range[0], x_range[1])
            y_shear = random.uniform(y_range[0], y_range[1])

            param_dict_.update({'x_shear': x_shear, 'y_shear': y_shear})

            return param_dict_

        elif aug_operation_ == 'gaussian_blur':
            sigma = random.uniform(self.aug_operations[aug_operation_][0], self.aug_operations[aug_operation_][1])
            param_dict_['sigma'] = sigma

            return param_dict_

        else:
            # This means there are no random parameters to generate for an augmentation transformation
            return None

    @staticmethod
    def augment_img(img_: np.ndarray, aug_operation_: str, operation_param_dict_: Optional[dict] = None) -> np.ndarray:

        if aug_operation_ == 'rotate_range':
            angle = operation_param_dict_['angle']
            (h, w) = img_.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
            rotated_img_ = cv.warpAffine(img_, M, (w, h))

            return rotated_img_

        elif aug_operation_ == 'horizontal_flip':
            img_ = cv.flip(img_, 1)
            return img_

        elif aug_operation_ == 'vertical_flip':
            img_ = cv.flip(img_, 0)
            return img_

        elif aug_operation_ == 'random_crop':
            h, w = img_.shape[0], img_.shape[1]
            img_ = img_[operation_param_dict_['h_start']: operation_param_dict_['h_end'],
                        operation_param_dict_['w_start']: operation_param_dict_['w_end']]
            img_ = cv.resize(img_, (h, w), interpolation=cv.INTER_AREA)
            return img_

        elif aug_operation_ == 'shearing':
            h, w = img_.shape[0], img_.shape[1]
            M2 = np.float32([[1, operation_param_dict_['x_shear'], 0],
                             [operation_param_dict_['y_shear'], 1, 0]])
            M2[0, 2] = -M2[0, 1] * h / 2
            M2[1, 2] = -M2[1, 0] * w / 2
            img_ = cv.warpAffine(img_, M2, (w, h))

            return img_

        elif aug_operation_ == 'gaussian_blur':
            filter_size = (0, 0)  # Size will be calculated based on the value of sigmaX
            img_ = cv.GaussianBlur(img_, filter_size,
                                   sigmaX=operation_param_dict_['sigma'], borderType=cv.BORDER_DEFAULT)
            return img_

        elif aug_operation_ == 'identity':
            return img_

        else:
            raise Exception('Augmented operation not programmed')

    @staticmethod
    def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
        pad_size = target_length - array.shape[axis]

        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


if __name__ == '__main__':
    data_path_source_dir = os.path.join('ml4h_proj1_colon_cancer_ct', 'ml4h_proj1_colon_cancer_ct')
    tr_df, x_ts_df = build_train_test_df(data_path_source_dir)

    cancer_pixels_df = pd.read_pickle('cancer_pixels_df')
    cancer_pixels_df.reset_index(inplace=True)
    cancer_pixels_df['index'] = cancer_pixels_df.image_name.map(
        lambda str_: str_.split('.nii.gz')[0].split('colon_')[1])

    tr_df_cancer_info = tr_df.join(
        cancer_pixels_df.set_index(['index', 'depth_i'])[['cancer_pixel_area']],
        how='left')
    tr_df_cancer_info['has_cancer_pixels'] = ~tr_df_cancer_info.cancer_pixel_area.isna()
    tr_df_cancer_info.cancer_pixel_area.fillna(0, inplace=True)

    resize_dim = (256, 256)

    # Test 2D Generator
    data_generator = DataGenerator2D(df=tr_df_cancer_info, x_col='x_tr_img_path', y_col='y_tr_img_path', batch_size=4,
                                     shuffle=True, shuffle_depths=True,
                                     resize_dim=resize_dim,
                                     class_sampling={'cancer_pixel': 2, 'not_cancer_pixel': 0.4},
                                     depth_class_col='has_cancer_pixels',
                                     rotate_range=30, horizontal_flip=True, vertical_flip=True,
                                     random_crop=(0.8, 0.9),
                                     shearing=((0.1, 0.3), (0., 0.0)), gaussian_blur=(0.3162, 0.9487))

    for X, y in data_generator:
        print(X.shape)
        print(y.shape)

        break

    # Test 2D Generator
    tr_3s_df = tr_df_cancer_info.reset_index(level=1).drop_duplicates(['x_tr_img_path', 'y_tr_img_path'], keep='last')\
                .loc[:, ['x_tr_img_path', 'y_tr_img_path', 'depth']]

    data_generator_3D = DataGenerator3D(df=tr_3s_df, x_col='x_tr_img_path', y_col='y_tr_img_path',
                                         batch_size=1,
                                         shuffle=False,
                                         resize_dim=resize_dim,
                                         rotate_range=30, horizontal_flip=True, vertical_flip=True,
                                         random_crop=(0.8, 0.9),
                                         shearing=((0.1, 0.3), (0., 0.0)), gaussian_blur=(0.3162, 0.9487))

    for i, (X, y) in enumerate(data_generator_3D):
        print(X.shape)
        print(y.shape)
        assert X.shape == y.shape
        #assert X.shape[3] == tr_3s_df.iloc[i]['depth'] + 1

        #if i == 10: break



