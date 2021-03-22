
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import cv2 as cv

from img_generator import DataGenerator2D


def calculate_iou(target: np.ndarray, prediction: np.ndarray) -> float:
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection.astype(np.float64)) / np.sum(union.astype(np.float64))

    return iou_score


def calculate_iou_holdout_set(holdout_df_: pd.DataFrame, img_dims: Tuple, model_,
                              pixel_threshold: float = 0.5, prediction_batch_size: int = 32) \
        -> Tuple[pd.DataFrame, list, list]:
    iou_list = list()
    y_ped_list = list()
    y_list = list()

    for img_dx, df_ in holdout_df_.groupby(level=0):
        img_i_generator = DataGenerator2D(df=df_, x_col='x_tr_img_path', y_col=None,
                                          batch_size=prediction_batch_size, num_classes=None, shuffle=False,
                                          resize_dim=img_dims)

        label_i_generator = DataGenerator2D(df=df_, x_col='x_tr_img_path', y_col='y_tr_img_path',
                                            batch_size=prediction_batch_size, num_classes=None, shuffle=False,
                                            resize_dim=None)

        # Predict for a group of cuts of the same image
        for i, ((X_cut_i, _), (_, y_cut_i)) in enumerate(zip(img_i_generator, label_i_generator)):

            y_cut_i_predict = model_.predict(X_cut_i)

            # Resize prediction to match label mask dimensions and restack
            #  the predictions so that hey are channel last
            for j, depth_i in enumerate(range(X_cut_i.shape[0])):
                y_cut_i_predict_resized_j = cv.resize(
                    y_cut_i_predict[j, :, :], y_cut_i.shape[1:],
                    interpolation=cv.INTER_CUBIC)  # INTER_LINEAR is faster but INTER_CUBIC is better

                # Add extra dim at the end
                y_cut_i_predict_resized_j = y_cut_i_predict_resized_j.reshape(y_cut_i_predict_resized_j.shape + (1,))
                y_cut_i_j = y_cut_i[j, :, :].reshape(y_cut_i[j, :, :].shape + (1,))

                if j == 0:
                    y_cut_i_predict_resized = y_cut_i_predict_resized_j
                    y_cut_i_restacked = y_cut_i_j

                else:
                    y_cut_i_predict_resized = np.concatenate([y_cut_i_predict_resized, y_cut_i_predict_resized_j],
                                                             axis=2)
                    y_cut_i_restacked = np.concatenate([y_cut_i_restacked, y_cut_i_j], axis=2)

            # When there is only one image in the minibatch it adds an extra dimension
            if len(y_cut_i_restacked.shape) > 3:
                y_cut_i_restacked = np.squeeze(y_cut_i_restacked, axis=3)

            # Now stack the minibatches along the 3rd axis to complete the 3D image
            if i == 0:
                y_i_predict_3d = y_cut_i_predict_resized
                y_i_3d = y_cut_i_restacked

            else:
                y_i_predict_3d = np.concatenate([y_i_predict_3d, y_cut_i_predict_resized], axis=2)
                y_i_3d = np.concatenate([y_i_3d, y_cut_i_restacked], axis=2)

        y_ped_list.append(y_i_predict_3d)
        y_list.append(y_i_3d)

        # Measure IoU over entire 3D image after concatenating all of the cuts
        iou_list.append({'index': img_dx,
                         'iou': calculate_iou(target=y_i_3d, prediction=(y_i_predict_3d > pixel_threshold) * 1)})

    # Let's convert the iou to a pandas dataframe
    iou_df = pd.DataFrame(iou_list).set_index('index')

    return iou_df, y_list, y_ped_list


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# Dice Loss
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

# Focal loss, alpha is the weight given
def focal_loss(gamma=2., alpha=.25):
    """

    :param gamma: In the papaer worked better with 2. It is an attenuation factor for the correct predictions of the
                    0 classes
    :param alpha: Alpha is used to specify the weight of different categories/labels, the size of the array needs to be
                    consistent with the number of classes.
                    If given only for one class, the for class 1 the weight is alpha, for class zero 1-alpha
    :return:
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed