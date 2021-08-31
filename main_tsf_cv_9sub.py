"""
use for CNN_basis.py
test the acc of 9 subject
cross validation data
"""


# from preprocess import *
# from CNN_basic import cnn

from preprocess_tsf_cv_9 import *
# from CNN_basic_tsf_cv_9 import cnn
# from easySTCNN import cnn
from CNN_for_tsf_9 import cnn,CNN
import os
from torchstat import stat
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,7'
devices = ['gpu:0,1,2,3,4']

subjects=[i+1 for i in range(1)]
# subjects.remove(9)
# sub_index = 2
for sub_index in subjects:
    # sub_index += 1
    # sub_index = 9
    cnn=CNN()
    stat(cnn,(1,16,1000))
    # data_train, label_train, data_test, label_test = split_subject(sub_index)
    # cnn(sub_index, data_train, label_train, data_test, label_test)




