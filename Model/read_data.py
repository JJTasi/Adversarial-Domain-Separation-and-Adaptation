import DataPackage as dp
import os
import sys
import numpy as np

def read(option):
    if option=='Amazon':
        #books electronics dvd kitchen
        max_feature=5000
        source_data, target_data = dp.datapackage(source='kitchen', target='books', max_feature=max_feature, tfidf_setting='seperate')
        
        return source_data, target_data, max_feature