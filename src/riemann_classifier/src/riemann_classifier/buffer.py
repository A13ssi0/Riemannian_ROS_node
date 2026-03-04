#!/usr/bin/env python3

import numpy as np
import rospy
from rosneuro_msgs.msg  import NeuroOutput
from std_msgs.msg import Float64MultiArray



def buffer_frames(eeg_lap, args):
    global buffered_eeg
    n_channels = args[0]
    fs = args[1]
    pub = args[2]
    
    eeg_lap.data = np.reshape(eeg_lap.data, (-1, n_channels))
    buffered_eeg = np.delete(buffered_eeg, [np.arange(eeg_lap.data.shape[0])], axis=0)
    buffered_eeg = np.append(buffered_eeg, eeg_lap.data, axis=0)
    tmp = buffered_eeg[~np.isnan(buffered_eeg[:, 0]), :]
    msg = Float64MultiArray()
    msg.data = tmp.flatten()
    pub.publish(msg)


if __name__ == '__main__':
    
    rospy.init_node('buffer', anonymous=False)

    fs = rospy.get_param('~samplerate')
    n_channels = rospy.get_param('~n_channels')
    classifier_type = rospy.get_param('~classifier_type')

    buffer_size = fs
        
    buffered_eeg = np.empty([buffer_size, n_channels])
    buffered_eeg.fill(np.nan)

    pub = rospy.Publisher('neurodata_buffered', Float64MultiArray, queue_size=1)
    rospy.Subscriber('neurodata_lap', Float64MultiArray, buffer_frames, (n_channels, fs, pub), queue_size=1)

    rospy.spin()




    

    
     