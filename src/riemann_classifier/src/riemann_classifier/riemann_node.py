#!/usr/bin/env python3

import rospy
import numpy as np
from rosneuro_msgs.msg  import NeuroOutput
from std_msgs.msg import Float64MultiArray
import py_utils.signal_processing as py_utils
from time import time
from json import dump
from datetime import datetime
from scipy.io import savemat
from pyriemann.utils.test import is_sym_pos_def
from py_utils.eeg_managment import get_channelsMask
from riemann_utils.covariances import center_covariance_online
from py_utils.signal_processing import RealTimeButterFilter



def classify_window(eeg, args):
    a = time()
    eeg_filt = filters_data(eeg, args)
    do_classification(eeg_filt, args)
    b = time()
    if (b-a) > 1/16:
        rospy.logwarn('Processing frequency: ' + str(1/(b-a)) + 'Hz')

def filters_data(eeg, args):
    n_channels = args[1]
    filters = args[4]

    eeg.data = np.reshape(eeg.data, (-1, n_channels))

    for filt in filters: 
        # print(f"[{self.name}] Applying filter: {filt}.")
        eeg.data = filt.filter(eeg.data)

    return eeg
    

def do_classification(eeg, args):

    classifier_dict = args[0]
    pub = args[3]

    prediction = NeuroOutput()
    
    eeg_lap = eeg.data @ classifier_dict['laplacian']
    prep_eeg = eeg_lap[:, classifier_dict['channelMask']]
    prep_eeg = np.expand_dims(prep_eeg, axis=0)
    if classifier_dict['normalizationMethod']=='trace':    cov = py_utils.get_covariance_matrix_traceNorm_online(prep_eeg)
    elif classifier_dict['normalizationMethod']=='lwf':      cov = py_utils.get_covariance_matrix_lwfNorm_online(prep_eeg)
    
    if classifier_dict['inv_sqrt_mean_cov'] is not None:
        cov = center_covariance_online(cov, classifier_dict['inv_sqrt_mean_cov'])
    if not (is_sym_pos_def(cov)): 
        print(f"[!!!] Covariance matrix is not SPD")  # for testing


    pred_proba = classifier_dict['fgmdm'].predict_probabilities(cov)
    
    prediction.softpredict.data = pred_proba.flatten()
    prediction.header.stamp = rospy.Time.now()
    prediction.decoder.classes = classifier_dict['classes']

    pub.publish(prediction)
    


if __name__ == '__main__':
    
    rospy.init_node('riemann_classifier', anonymous=False)

    n_channels = rospy.get_param('~n_channels')
    subject = rospy.get_param('~subject')

    path_classifier = rospy.get_param('~decoder_path')
    print('Loading classifier from: ' + path_classifier)
    classifier_dict = py_utils.data_managment.load(path_classifier)
    print('Classifier loaded')

    bandPassFreqs = classifier_dict['bandPass']
    stopBandFreqs = classifier_dict['stopBand']

    filters = []
    if len(bandPassFreqs)>0:
        print(f" - Applying bandpass filter: {bandPassFreqs[0]} Hz")
        filters.append(RealTimeButterFilter(classifier_dict['filter_order'], np.array(bandPassFreqs[0]), classifier_dict['fs'], 'bandpass'))

    if len(stopBandFreqs)>0:
        print(f" - Applying stopband filter: {stopBandFreqs[0]} Hz")
        filters.append(RealTimeButterFilter(classifier_dict['filter_order'], np.array(stopBandFreqs[0]), classifier_dict['fs'], 'bandstop'))

    pub = rospy.Publisher('/smr/neuroprediction/raw', NeuroOutput, queue_size=1)
    rospy.Subscriber('neurodata_buffered', Float64MultiArray, classify_window, (classifier_dict, n_channels,  pub, filters), queue_size=16)
    
    rospy.spin()





    

    
     