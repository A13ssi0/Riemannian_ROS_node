#!/usr/bin/env python3

import numpy as np
import rospy
import utils as utils
from rosneuro_msgs.msg  import NeuroOutput
from std_msgs.msg import Float64MultiArray
from scipy.signal import lfilter



def get_logBP(buffered_eeg, args):

    fs = args[0]
    n_channels = args[1]
    pub = args[2]
    

    squared =  np.reshape(buffered_eeg.data, (-1, n_channels)) **2
    # wait until buffer is full
    if squared.shape[0] == 1024 :
        # moving avg (1 s win)
        avg = lfilter(np.full((int(fs)), 1/fs), 1, squared, axis=0)
        avg = avg[512:, :]
        log = np.log10(avg)

        msg = Float64MultiArray()
        msg.data = log.flatten()
        pub.publish(msg)


if __name__ == '__main__':
    
    rospy.init_node('logBP', anonymous=False)

    fs = rospy.get_param('~samplerate')
    n_channels = rospy.get_param('~n_channels')
    classifier_type = rospy.get_param('~classifier_type')

    pub = rospy.Publisher('neurodata_logBP', Float64MultiArray, queue_size=1)

    if classifier_type=='fgmdm_logBP':
        rospy.Subscriber('neurodata_buffered', Float64MultiArray, get_logBP, (fs, n_channels, pub), queue_size=1)

    rospy.spin()




    

    
     