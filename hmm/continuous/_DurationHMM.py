'''
Created on Oct 31, 2014

@author: joro
'''
import numpy
import os
import sys

from numpy.core.numeric import Infinity

from _ContinuousHMM import _ContinuousHMM
from hmm.continuous.DurationPdf  import DurationPdf, MINIMAL_PROB
PATH_LOGS = '.'

parentDir = os.path.abspath(  os.path.join(os.path.dirname(os.path.realpath(sys.argv[0]) ), os.path.pardir,  os.path.pardir ) ) 
pathUtils = os.path.join(parentDir, 'utilsLyrics')
if pathUtils not in sys.path: sys.path.append(pathUtils )

from Utilz import writeListOfListToTextFile, writeListToTextFile

# PATH_LOGS='/Users/joro/Downloads/'
# PATH_LOGS='.'



# ALPHA =  0.99
OVER_MAX_DUR_FACTOR = 1.3

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class _DurationHMM(_ContinuousHMM):
    '''
    Implements the decoding with duration probabilities, but should not be used directly.
    '''
    
    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=numpy.double, verbose=False):
            '''
            See _ContinuousHMM constructor for more information
            '''
            _ContinuousHMM.__init__(self,n,m,d,A,means,covars,w,pi,min_std,init_type,precision,verbose) #@UndefinedVariable
            
    def setALPHA(self, ALPHA):
        # DURATION_WEIGHT 
        self.ALPHA = ALPHA
    
    def setDurForStates(self, listDurations):
        '''
        mapping of state to its duration (in number of frames).
        @param listDurations read from  state.Durations
        
        '''
        if len(listDurations) != self.n:
            sys.exit("count of fiven Durations diff from map")
            
        self.durationMap =  numpy.array(listDurations, dtype=int)
        # DEBUG: 
        writeListToTextFile(self.durationMap, None , PATH_LOGS + '/durationMap') 

#         STUB
#         self.durationMap =  numpy.arange(1,self.n+1)
       
        # set duration lookup table
        self.MAX_DUR = int(numpy.amax(self.durationMap))
        self.durationPdf = DurationPdf(self.MAX_DUR, self.usePersistentFiles)

    def getWaitLogLik(self, d, state):
        '''
        return waiting pdf. uses normal distribution
        IMPORTANT: if d>D function should still return values up to some limit (e.g. +100% and least till MaxDur)
        STUB
        '''  
        
        scoreDurCurrState = self.durationMap[state]
        
        return self.durationPdf.getWaitLogLik(d, scoreDurCurrState)
         
    
    def _viterbiForcedDur(self, observations):
        # sanity check. make sure durations are init from score
        try: 
            self.durationMap
        except NameError:
            sys.exit(NameError.message)
        
        print "loading probs all observations"
        self._mapB(observations)
        
        print "decoding..."
        
       
        # backpointer: how much duration waited in curr state  
        self.chi = numpy.empty((len(observations),self.n),dtype=self.precision)
        self.chi.fill(-1)
        
         # backpointer: prev. state  
        self.psi = numpy.empty((len(observations),self.n),dtype=self.precision)
        self.psi.fill(-1)
        
        # init. t< MAX_DUR
        self._initBeginingPhis(len(observations))
        
        if (self.MAX_DUR>= len(observations)):
            sys.exit("MAX_Dur {} of a state is more than total number of obesrvations {}. Unable to decode".format(self.MAX_DUR, len(observations)) )
        
        for t in range(self.MAX_DUR,len(observations)):
            for currState in xrange(self.n):
                self._calcCurrStatePhi(t, currState) # get max duration quantities
        
        writeListOfListToTextFile(self.phi, None , PATH_LOGS + '/phi') 
            
        # return for backtracking
        return  self.chi, self.psi
    
    
    def _calcCurrStatePhi(self,  t, currState):
        '''
        calc. quantities in recursion  equation
        '''
        logger.info("at time t={}".format(t) )          

        currMaxDur = self.durationMap[currState]
        # take 30% more from dur from score 
        currMaxDur = int(round(OVER_MAX_DUR_FACTOR * currMaxDur))
       
        maxPhi, fromState,  maxDurIndex =  self.getMaxPhi(t, currState, currMaxDur)
        
        
                
        self.phi[t][currState] = maxPhi
        
        self.psi[t][currState] = fromState
        
        self.chi[t][currState] = maxDurIndex
                
           
    def _initBeginingPhis(self, lenObservations):
        '''
        initi phis when t < self.MAX_DUR
        '''
        self._initKappas()
        
         # for convenience put as class vars
        self.phi = numpy.empty((lenObservations,self.n),dtype=self.precision)
        self.phi.fill(-Infinity)
        
        
        # init t=0
#         for currState in range(self.n): self.phi[0,currState] = self.kappas[currState,0]
        self.phi[0,:] = self.kappas[0,:]
        
        # init first currState. done to allow self.getMaxPhi  to access prev. currState
        self.phi[:len(self.kappas[:,0]),0] = self.kappas[:,0]        
        
        # select max (kappa and phi_star)
      
        for t in  range(1,int(self.MAX_DUR)):
            logger.info("at time t={}".format(t) )          
            for currState in range(1, self.n): 
                
                currMaxDur = self.durationMap[currState]
                # take 30% more from dur from score 
                currMaxDur = int(round(OVER_MAX_DUR_FACTOR * currMaxDur))
                
                currReducedMaxDur = min(t, currMaxDur)
                phiStar, fromState,  maxDurIndex  =  self.getMaxPhi(t, currState, currReducedMaxDur)
                
                if  phiStar > self.kappas[t,currState] :
                    self.phi[t,currState] = phiStar
                    self.psi[t,currState] = fromState
                    self.chi[t,currState] = maxDurIndex
                else:
                    logger.debug( " kappa more than phi at time {} and state {}".format(t, currState))                        
                    self.phi[t, currState] = self.kappas[t, currState]
                    # kappas mean still at beginning state
                    self.psi[t,currState] = currState
                    self.chi[t,currState] = t
                    
        
        writeListOfListToTextFile(self.phi, None , PATH_LOGS + '/phi_init') 

        
    def _initKappas(self):
        '''
        kappas[t][s] - starting and staying at time t in same currState s.
        WITH LogLik 
        '''
        print 'init kappas...'
        self.kappas = numpy.empty((self.MAX_DUR,self.n), dtype=self.precision)
        # if some kappa[t, state] = -INFINITY and phi[t,state] = -INFINITY, no initialization is possilbe
        self.kappas.fill(numpy.log(MINIMAL_PROB))
        
        for currState in range(self.n):
            sumObsProb = 0
            currDmax = self.durationMap[currState]
            currLogPi = numpy.log(self.pi[currState])
            
            for d in range(1,int(currDmax)+1):
                                
                updateQuantity, sumObsProb = self._calcUpdateQuantity(d-1, d, currState, 0, sumObsProb)
                
                self.kappas[d-1,currState] = currLogPi + updateQuantity
                
                 #sanity check
                if self.kappas[d-1,currState] == 0:
                     print "underflow error at time {}, currState {}".format(d-1, currState)
                
                
        writeListOfListToTextFile(self.kappas, None , PATH_LOGS + '/kappas') 
    
    
  
        
        
    def getMaxPhi(self, t, currState, maxPossibleDuration):
        '''
        recursive rule. Find duration that maximizes current phi
        @return: maxPhi - pprob
        @return: fromState - from which state we come (hard coded to prev. state in forced alignment) 
        @return: maxDurIndex - index Duration with max prob. INdex for t begins at 0
        
        used in _initBeginingPhis
        used in _calcCurrStatePhi
        '''
        sumObsProb = 0
         # due to forced alignment
        fromState = currState - 1
        maxPhi = -1 * numpy.Infinity 
        maxDurIndex = -1
        
#         print "in getMaxPhi: maxDuration =",  maxPossibleDuration
          
        for d in range(1, maxPossibleDuration+1):

            currPhi = self.phi[t-d][fromState]
                        
            updateQuantity, sumObsProb = self._calcUpdateQuantity(t-d+1, d, currState, currPhi, sumObsProb)
            #sanity check

            if updateQuantity > maxPhi:
                maxPhi = updateQuantity
                maxDurIndex = d
        
        if maxDurIndex == -1:
            sys.exit(" no max duration at time {} and state {}".format(t, currState))
        return maxPhi, fromState, maxDurIndex    
    
    def _calcUpdateQuantity(self, whichTime, whichDuration, currState, currPhi, sumObsProb):
        '''
        calc update quantity.
        used in getMaxPhi
        used in init kappas
        '''
        
    #       print " d= {} time = {}, state = {}".format(whichDuration, whichTime, currState ) 

#             print "\t\t currPhi= {}".format(currPhi)  

        
        waitLogLik = self.getWaitLogLik(whichDuration, currState)
#         print  "\t\t waitLogLik= {}".format (waitLogLik) 
            
        sumObsProb += self.B_map[currState, whichTime]
#         print "\t\t sumObsProb= {}".format( sumObsProb)      
        
        updateQuantity = currPhi + self.ALPHA * waitLogLik + (1-self.ALPHA)*sumObsProb
#         updateQuantity = currPhi +  waitLogLik + sumObsProb
#             print "\t UPDATE QUANT= {}".format(updateQuantity)  


        return updateQuantity, sumObsProb
             
