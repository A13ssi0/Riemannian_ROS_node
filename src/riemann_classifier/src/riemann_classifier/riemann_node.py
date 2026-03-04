#!/usr/bin/env python3


import rospy
import numpy as np
from rosneuro_msgs.msg  import NeuroOutput
from std_msgs.msg import Float64MultiArray
import utils as utils
from time import time
from json import dump
from datetime import datetime
from scipy.io import savemat
from pyriemann.utils.test import is_sym_pos_def



def classify_window(eeg_lap, args):
    a = time()
    do_classification(eeg_lap, args)
    b = time()
    if (b-a) > 1/16:
        rospy.logwarn('Processing frequency: ' + str(1/(b-a)) + 'Hz')
        timestamp = (datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' Processing frequency: ' + str(1/(b-a)) + 'Hz')
        with open(filepath + 'LOG_' + subject + '_'+ today + '.txt', 'a') as filehandle:
            dump(timestamp, filehandle)
            filehandle.write(',\n')


def do_classification(eeg_lap, args):

    # global all_covs
    # global data_covs
    # global all_unfilt

    classifier_dict = args[0]
    n_channels = args[1]
    classifier_type = args[2]
    pub = args[3]

    eeg_lap.data = np.reshape(eeg_lap.data, (-1, n_channels))

    prediction = NeuroOutput()
    
    classifier = classifier_dict[classifier_type]

    eeg = eeg_lap.data
    [eeg, channels] = utils.select_channels(eeg_lap.data, classifier_dict['wantedChannels'])
    #eeg = utils.apply_ROI_over_channels(eeg, channels, classifier_dict['channelGroups'])
    #eeg_unfilt = np.expand_dims(eeg, axis=0)
    eeg = np.expand_dims(eeg, axis=0)
    #eeg = utils.get_bandranges(eeg, classifier_dict['bandranges'], classifier_dict['fs'], classifier_dict['filter_order'], []) # classifier_dict['stoprange'])
    #print(eeg.shape)
    cov, data_cov = utils.get_covariance_matrix_traceNorm_online(eeg)
    #print(cov.shape)
    # center cov matrices wrt ref matrix used for the classifier creation
    # cov_centered = utils.center_covariance_online(cov, classifier_dict['inv_sqrt_mean_cov'])
    cov_centered = cov
    
    # if data_cov.shape[1]==512:
    #     data_covs = np.append(data_covs, data_cov, axis=0)
    #     all_covs = np.append(all_covs, cov_centered, axis=0)
    #     #all_unfilt =  np.append(all_unfilt, eeg_unfilt, axis=0)

    if not is_sym_pos_def(cov_centered):
        # pass
        print('not SPD')
        val, _ = np.linalg.eig(np.squeeze(cov_centered))
        print(val)
        pred_proba = np.empty((1,1,2))
        pred_proba[0,0] = np.array([0.5, 0.5])
    else:
        pred_proba = classifier.predict_probabilities(cov_centered)
     
    # pred = classifier.merge_classifiers(pred_proba)[0]
    # EDITED
    # pred = classifier.merge_bands(pred_proba)[0]
    #pred = pred/np.sum(pred)
    prediction.softpredict.data = pred_proba.flatten()
    prediction.header.stamp = rospy.Time.now()
    prediction.decoder.classes = classifier_dict['classes']

    # if all_covs.shape[0]==600 :
    #     scipy.io.savemat('/home/mtld/mtld_workspace/'+ subject+'_'+today+'_ros_covs.mat', {'ros_covs':all_covs, 'ros_data': data_covs})#, 'data_unfilt': all_unfilt})


    pub.publish(prediction)
    


if __name__ == '__main__':
    
    rospy.init_node('riemann_classifier', anonymous=False)

    filepath = '/home/mtld/mtld_workspace/logs/'
    n_channels = rospy.get_param('~n_channels')
    subject = rospy.get_param('~subject')
    classifier_type = rospy.get_param('~classifier_type')

    # all_covs = np.empty([1, 1, 8, 8])
    # data_covs = np.empty([1, 512, 8])
    # all_unfilt = np.empty([1, 512, 8])

    path_classifier = rospy.get_param('~decoder_path')
    print('Loading classifier from: ' + path_classifier)
    classifier_dict = utils.load(path_classifier)
    print('Classifier loaded')
    #pub = rospy.Publisher('riemann_output', NeuroOutput, queue_size=1)
    # EDITED
    pub = rospy.Publisher('/smr/neuroprediction/raw', NeuroOutput, queue_size=1)
    if classifier_type=='fgmdm':
        rospy.Subscriber('neurodata_buffered', Float64MultiArray, classify_window, (classifier_dict,n_channels, classifier_type, pub), queue_size=16)
    if classifier_type=='fgmdm_logBP':
        rospy.Subscriber('neurodata_logBP', Float64MultiArray, classify_window, (classifier_dict,n_channels, classifier_type, pub), queue_size=16)
    
    today = datetime.now()
    today = today.strftime("%Y%m%d.%H%M%S")

    rospy.spin()





# def classify_window(eeg_lap, args):

#     classifier_dict = args[0]
#     n_channels = args[1]
#     pub = args[2]
    
#     eeg_lap.data = np.reshape(eeg_lap.data, (-1, n_channels))

#     # start_time = time()


#     prediction = NeuroOutput()
    
#     classifier = classifier_dict['fgmdm']

#     eeg = utils.select_channels(eeg_lap.data,classifier_dict['wantedChannels'])
#     eeg = utils.get_bandranges(eeg, classifier_dict['bandranges'], classifier_dict['fs'], classifier_dict['filter_order'])
    
#     cov = utils.get_covariance_matrix_traceNorm_online(eeg)
#     cov_centered = utils.center_covariance_online(cov, classifier_dict['inv_sqrt_mean_cov'])

#     prediction.softpredict.data = classifier.predict_probabilities_mergeClassifier(cov_centered)
#     prediction.header.stamp = rospy.Time.now()
#     pub.publish(prediction)

#     # if (1/(time() - start_time)) < (20):
#     #     rospy.logwarn('Processing frequency: ' + str(1/(time() - start_time)) + 'Hz')

    

    
     