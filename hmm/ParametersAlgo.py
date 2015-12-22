'''
Created on May 28, 2015

@author: joro
'''

######### PARAMS:
class ParametersAlgo(object):
    THRESHOLD_PEAKS = -70

    DEVIATION_IN_SEC = 2

    # unit: num frames
    NUMFRAMESPERSECOND = 100
    
    CONSONANT_DURATION_IN_SEC = 0.3
    CONSONANT_DURATION = NUMFRAMESPERSECOND * CONSONANT_DURATION_IN_SEC;
    
    CONSONANT_DURATION_DEVIATION = 0.7
    
    