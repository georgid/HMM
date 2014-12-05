'''
Created on Nov 6, 2014

@author: joro
'''

from scipy.stats import norm
from numpy import linspace
import sys
from numpy.core.arrayprint import set_printoptions
import numpy
from numpy.core.numeric import Infinity
import os

PATH_LOGS = '.'

# to replace 0: avoid log(0) = -inf. -Inf + p(d) makes useless the effect of  p(d)
MINIMAL_PROB = sys.float_info.min

parentDir = os.path.abspath(  os.path.join(os.path.dirname(os.path.realpath(sys.argv[0]) ), os.path.pardir,  os.path.pardir, os.path.pardir ) ) 
pathUtils = os.path.join(parentDir, 'utilsLyrics')
if pathUtils not in sys.path: sys.path.append(pathUtils )

from Utilz import writeListOfListToTextFile

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DurationPdf(object):
    '''
    classdocs
    '''

    def __init__(self,  MAX_DUR, usePersistentProbs,  distributionType=1):
        
        # 1 - normal. 
        # 2 - gamma TODO: implement
        self.distributionType = distributionType
        
        '''
        maxDur x currDur lookupTable of probs
        '''
        self.MAX_DUR = MAX_DUR
        '''
        how much of a phoneme may be longer than its max_dur 
        '''
        if distributionType == 1:
            self.MAX_ALLOWED_DURATION_RATIO = 2
            
        self.lookupTableLogLiks  = numpy.empty((MAX_DUR, self.MAX_ALLOWED_DURATION_RATIO * MAX_DUR + 1))
        self.lookupTableLogLiks.fill(-Infinity)
        
        self.minVal = norm.ppf(0.01)
        self.maxVal= norm.ppf(0.99)
        
        self._constructLogLiksTable(usePersistentProbs)
       
   
    
    def _constructLogLiksTable(self, usePersistentProbs):
        
        PATH_LOOKUP_DUR_TABLE = PATH_LOGS + '/lookupTable'
        if usePersistentProbs and os.path.exists(PATH_LOOKUP_DUR_TABLE): 
            self.lookupTableLogLiks = numpy.loadtxt(PATH_LOOKUP_DUR_TABLE)
            
            # if table covers max dur
            if self.lookupTableLogLiks.shape[0] >= self.MAX_DUR:
#                 sys.exit('not enough durations in loaded table. Max should be {}  delete table {} and generate them again'.format(self.MAX_DUR, PATH_LOOKUP_DUR_TABLE))
                return   
        
        # construct
        else:
            logging.info("constructing duration Probability lookup Table...")

            for currMaxDur in range(1,int(self.MAX_DUR)+1):
                self._constructLogLikDistrib( currMaxDur)
            if usePersistentProbs:
                writeListOfListToTextFile(self.lookupTableLogLiks, None ,  PATH_LOOKUP_DUR_TABLE) 

            
        
    def _constructLogLikDistrib(self, currMaxDur):
        '''
        calc and store logLiks for given @param currMaxDur
        range currMaxDur is (1,currMaxDur)
        '''
        
         #get min and max
     
        numBins = self.MAX_ALLOWED_DURATION_RATIO * currMaxDur + 1
        quantileVals  = linspace(self.minVal, self.maxVal, numBins)
        
        for d in range(1,numBins):
            lik = norm.pdf(quantileVals[d])
            self.lookupTableLogLiks[currMaxDur-1,d] = lik
        
        # normalize all liks to sum to 1
#         self.lookupTableLogLiks[currMaxDur-1,1:numBins] = _ContinuousHMM._normalize( self.lookupTableLogLiks[currMaxDur-1,1:numBins]) 
            
        logging.debug("sum={} for max dur {}".format(sum(self.lookupTableLogLiks[currMaxDur-1,1:numBins]), currMaxDur))
        
        self.lookupTableLogLiks[currMaxDur-1,1:numBins] = numpy.log( self.lookupTableLogLiks[currMaxDur-1,1:numBins] )
            
            
            
    def getWaitLogLik(self, d, scoreDur):
        '''
        get lik for duration d for given score duration scoreDur for phoneme  
        used in _DurationHMM
        '''
        
        ##### make sure zero is returned
       
        
        if d==0:
            sys.exit("d = 0 not implemented yet")
            return 
        # used in kappa. -Inf because we never want kappa to be selected if over max region of duration
        elif d >= self.MAX_ALLOWED_DURATION_RATIO * scoreDur + 1:
            return -Infinity
        else:
            if scoreDur > self.lookupTableLogLiks.shape[0]:
                sys.exit("".format(self.lookupTableLogLiks.shape[0], scoreDur))
            return self.lookupTableLogLiks[scoreDur-1,d] 
#         set_printoptions(threshold='nan') 
    
    
if __name__ == '__main__':
    durPdf = DurationPdf(30, True)
    
    print durPdf.lookupTableLogLiks
    
    print durPdf.getWaitLogLik(10, 10)
    
    
        
        