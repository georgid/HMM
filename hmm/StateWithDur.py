'''
Created on Nov 10, 2014

@author: joro
'''
import os
import sys
parentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__) ), os.path.pardir, os.path.pardir)) 

htkModelParser = os.path.join(parentDir, 'htkModelParser')


sys.path.append(htkModelParser )
from htk_models import State

class StateWithDur(State):
    '''
    extends State with 
    - durationInMinUnit (in minimal_duration unit)
    - durationInMinUnit (in Frames)
    '''


    def __init__(self, mixtures, phonemeName, idxInPhoneme, distribType='normal'):
        '''
        Constructor
        '''
        State.__init__(self, mixtures)
        self.phonemeName = phonemeName
        self.idxInPhoneme  = idxInPhoneme
        
        try:
            distribType
        except NameError:
            pass
        else:
            if not distribType=='normal' and not distribType=='exponential':
                sys.exit(" unknown distrib type. Only normal and exponential aimplemented now!")
            
        self.distributionType = distribType
                                
  
    def setDurationInFrames(self, durationInFrames):
        self.durationInFrames = durationInFrames
        
    def getDurationInFrames(self):
        
        try:  
            return self.durationInFrames
        except AttributeError:
            return 0
        
    def setWaitProb(self, waitProb):   
        self.waitProb = waitProb
 
    
    def __str__(self):
        return self.phonemeName + "_"  + str(self.idxInPhoneme)
        
        