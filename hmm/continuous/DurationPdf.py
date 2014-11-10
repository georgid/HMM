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


class DurationPdf(object):
    '''
    classdocs
    '''

    def __init__(self,  MAX_DUR, distributionType=1):
        
        # 1 - normal. 
        # 2 - gamma TODO: implement
        self.distributionType = distributionType
        
        '''
        maxDur x currDur lookupTable of probs
        '''
        self.MAX_DUR = MAX_DUR
        self.lookupTableLogLiks  = numpy.empty((MAX_DUR, 2 * MAX_DUR + 1))
        self.lookupTableLogLiks.fill(-Infinity)
        
        self.minVal = norm.ppf(0.01)
        self.maxVal= norm.ppf(0.99)
        
        self._constructLogLiksTable()
       
   
    
    def _constructLogLiksTable(self):
        for currMaxDur in range(1,int(self.MAX_DUR)+1):
            self._constructLogLikDistrib( currMaxDur)
            
        
    def _constructLogLikDistrib(self, maxDur):
        '''
        calc and store logLiks for given @param maxDur
        range maxDur is (1,maxDur)
        '''
        
         #get min and max
     
        numBins = 2*maxDur + 1
        quantileVals  = linspace(self.minVal, self.maxVal, numBins)
        
        for d in range(1,numBins):
            logLik = numpy.log( norm.pdf(quantileVals[d]))
            self.lookupTableLogLiks[maxDur-1,d] = logLik
            
            
            
    def getWaitLogLik(self, d, maxDur):
        '''
        used in _DurationHMM
        '''
        
        ##### make sure zero is returned
       
        
        if d==0:
            sys.exit("d = 0 not implemented yet")
            return 
        
        elif d >= 2*maxDur + 1:
            return -Infinity
        else:
            return self.lookupTableLogLiks[maxDur-1,d] 
#         set_printoptions(threshold='nan') 
    
    
if __name__ == '__main__':
    durPdf = DurationPdf(10)
    
    print durPdf.lookupTableLogLiks
    
    print durPdf.getWaitLogLik(10, 10)
    
    
        
        