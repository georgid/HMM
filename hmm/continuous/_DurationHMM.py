'''
Created on Oct 31, 2014

@author: joro
'''
import numpy
import os
import sys

from numpy.core.numeric import Infinity

from _ContinuousHMM import _ContinuousHMM
from hmm.continuous.DurationPdf  import DurationPdf, \
    deviationInSec, NUMFRAMESPERSEC
from hmm.continuous.ExpDurationPdf import ExpDurationPdf

# to replace 0: avoid log(0) = -inf. -Inf + p(d) makes useless the effect of  p(d)
MINIMAL_PROB = sys.float_info.min


parentDir = os.path.abspath(  os.path.join(os.path.dirname(os.path.realpath(sys.argv[0]) ), os.path.pardir,  os.path.pardir ) ) 
pathUtils = os.path.join(parentDir, 'utilsLyrics')
if pathUtils not in sys.path: sys.path.append(pathUtils )

from Utilz import writeListOfListToTextFile, writeListToTextFile

# PATH_LOGS='/Users/joro/Downloads/'
PATH_LOGS='.'



# ALPHA =  0.99
# OVER_MAX_DUR_FACTOR = 1.3



class _DurationHMM(_ContinuousHMM):
    '''
    Implements the decoding with duration probabilities, but should not be used directly.
    '''
    def __init__(self,statesNetwork, numMixtures, numDimensions):
    
#     def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=numpy.double, verbose=False):
            '''
            See _ContinuousHMM constructor for more information
            '''
            means, covars, weights, pi = self._constructHMMNetworkParameters(statesNetwork, numMixtures, numDimensions)
             
            n = len(statesNetwork)
            min_std=0.01
            init_type='uniform'
            precision=numpy.double
            verbose = False 
            _ContinuousHMM.__init__(self, n, numMixtures, numDimensions, None, means, covars, weights, pi, min_std,init_type,precision,verbose) #@UndefinedVariable
    
            self.statesNetwork = statesNetwork
            self.setDurForStates(listDurations=[])
      
    def _constructHMMNetworkParameters(self,  statesSequence, numMixtures, numDimensions):
        '''
        tranform other htkModel params to  format of gyuz's hmm class
        '''
        
       
        numStates = len(statesSequence)
        means = numpy.empty((numStates, numMixtures, numDimensions))
        
        # init covars
        covars = [[ numpy.matrix(numpy.eye(numDimensions,numDimensions)) for j in xrange(numMixtures)] for i in xrange(numStates)]
        
        weights = numpy.ones((numStates,numMixtures),dtype=numpy.double)
        
        # start probs :
        pi = numpy.zeros((numStates), dtype=numpy.double)
        
        # avoid log(0) 
        pi.fill(sys.float_info.min)
#          allow to start only at first state
        pi[0] = 1
        
        # equal prob. for states to start
#         pi = numpy.ones( (numStates)) *(1.0/numStates)
        
    
         
        if statesSequence==None:
            sys.exit('no state sequence')
               
        for i in range(len(statesSequence) ):
            state  = statesSequence[i] 
            
            for (numMixture, weight, mixture) in state.mixtures:
                
                weights[i,numMixture-1] = weight
                
                means[i,numMixture-1,:] = mixture.mean.vector
                
                variance_ = mixture.var.vector
                for k in  range(len( variance_) ):
                    covars[i][numMixture-1][k,k] = variance_[k]
        return means, covars, weights, pi
    
                
    def setALPHA(self, ALPHA):
        # DURATION_WEIGHT 
        self.ALPHA = ALPHA
    

    
    def setDurForStates(self, listDurations):
        '''
        mapping of state to its duration (in number of frames).
        @param listDurations read from  state.Durations
        
        '''
        if listDurations == []:
            for  stateWithDur_ in self.statesNetwork:
                listDurations.append(stateWithDur_.getDurationInFrames())
                
        # sanity check  
        if len(listDurations) != self.n:
            sys.exit("#Durations from list = {}, whereas #states={}".format(len(listDurations), self.n ))

        self.durationMap =  numpy.array(listDurations, dtype=int)
        # DEBUG: 
        writeListToTextFile(self.durationMap, None , PATH_LOGS + '/durationMap') 

#         STUB
#         self.durationMap =  numpy.arange(1,self.n+1)
       
#         self.durationPdf = DurationPdf(self.R_MAX, self.usePersistentFiles)
        self.durationPdf = DurationPdf()
        self.R_MAX = int( self.durationPdf.getEndDur(numpy.amax(self.durationMap) ) )
    


    def getWaitLogLik(self, d, whichState):
        '''
        return waiting pdf. uses normal distribution
        IMPORTANT: if d>D function should still return values up to some limit (e.g. +100% and least till MaxDur)
        STUB
        '''  
        stateWithDuration = self.statesNetwork[whichState]
        if stateWithDuration.distributionType=='exponential':
            durationDistrib = ExpDurationPdf(stateWithDuration.waitProb)
            return durationDistrib.getWaitLogLik(d)
        

        scoreDurCurrState = self.durationMap[whichState]
        return self.durationPdf.getWaitLogLik(d, scoreDurCurrState)
         
    
    def _viterbiForcedDur(self, observations):
        # sanity check. make sure durations are init from score
      
        
        print "decoding..."
        for t in range(self.R_MAX,len(observations)):                          
            for currState in xrange(self.n):
                self._calcCurrStatePhi(t, currState) # get max duration quantities
        
        writeListOfListToTextFile(self.phi, None , PATH_LOGS + '/phi') 
            
        # return for backtracking
        return  self.chi, self.psi
    

    
    def initDecodingParameters(self, observations):
        '''
        helper method to init all params
        '''
        lenObservations = len(observations)
        try: 
            self.durationMap
        except NameError:
            sys.exit(NameError.message)
        
        self._mapB(observations)
    
        # backpointer: how much duration waited in curr state
        self.chi = numpy.empty((lenObservations, self.n), dtype=self.precision)
        self.chi.fill(-1)
       
        # backpointer: prev. state
        self.psi = numpy.empty((lenObservations, self.n), dtype=self.precision)
        self.psi.fill(-1)
   
        # init. t< R_MAX
        self._initBeginingPhis(lenObservations)
        if (self.R_MAX >= lenObservations):
            sys.exit("MAX_Dur {} of a state is more than total number of observations {}. Unable to decode".format(self.R_MAX, lenObservations))
    
    
    def _calcCurrStatePhi(self,  t, currState):
        '''
        calc. quantities in recursion  equation
        '''
        self.logger.debug("at time t={}".format(t) )          

        currRefDur = self.durationMap[currState]
        # take 30% more from dur from score 
#         currRefDur = int(round(OVER_MAX_DUR_FACTOR * currRefDur))
       
        minDur = self.durationPdf.getStartDur(currRefDur)
        endDur =  self.durationPdf.getEndDur(currRefDur)
        maxPhi, fromState,  maxDurIndex =  self.getMaxPhi(t, currState, minDur, endDur)
        
        
                
        self.phi[t][currState] = maxPhi
        
        self.psi[t][currState] = fromState
        
        self.chi[t][currState] = maxDurIndex
                
           

    def computePhiStar(self, t, currState):
        '''
        boundaries check for minDur and endDur makes PhiStar different from phi 
        '''
        
        fromState =-1
        maxDurIndex = -1
        
        currRefDur = self.durationMap[currState] # take 30% more from dur from score
    #                 currRefDur = int(round(OVER_MAX_DUR_FACTOR * currRefDur))
        
        ####### boundaries check
        minDur = self.durationPdf.getStartDur(currRefDur)
        if t <= minDur: # min duration is before beginning of audio 
            phiStar = -Infinity
        else:
            currReducedMaxDur = min(t, self.durationPdf.getEndDur(currRefDur))
            phiStar, fromState, maxDurIndex = self.getMaxPhi(t, currState, minDur, currReducedMaxDur)
        return phiStar, fromState, maxDurIndex
    

    def _initBeginingPhis(self, lenObservations):
        '''
        init phis when t < self.R_MAX
        '''
        
        self._initKappas(lenObservations)
        
         # for convenience put as class vars
        self.phi = numpy.empty((lenObservations,self.n),dtype=self.precision)
        self.phi.fill(-Infinity)
        
        
        # init t=0
#         for currState in range(self.n): self.phi[0,currState] = self.kappas[currState,0]
        self.phi[0,:] = self.kappas[0,:]
        
        # init first state = kappa (done to allow self.getMaxPhi  to access prev. currState)
        self.phi[:len(self.kappas[:,0]),0] = self.kappas[:,0]        
        
      
        # select bigger (kappa and phi_star)
        for t in  range(1,int(self.R_MAX)):
            self.logger.debug("at time t={}".format(t) )          
            # phi start makes sence only from second state 
            for currState in range(1, self.n): 
                
                phiStar, fromState, maxDurIndex = self.computePhiStar(t, currState)
                
                # take bigger : eq:deltaStarOrKappa
                if  phiStar > self.kappas[t,currState] :
                    self.phi[t,currState] = phiStar
                    self.psi[t,currState] = fromState 
                    self.chi[t,currState] = maxDurIndex
                else:
                    self.logger.debug( " kappa more than phi at time {} and state {}".format(t, currState))                        
                    self.phi[t, currState] = self.kappas[t, currState]
                    # kappas mean still at beginning state
                    self.psi[t,currState] = currState
                    self.chi[t,currState] = t
                    
        
        writeListOfListToTextFile(self.phi, None , PATH_LOGS + '/phi_init') 

        
    def _initKappas(self, lenObservations):
        '''
        kappas[t][s] - starting and staying at time t in same currState s.
        WITH LogLik 
        '''
        if lenObservations <= self.R_MAX:
            sys.exit("observations are only {}, R_max = {}. not able to run initialization. Increase size of observations".format(lenObservations,self.R_MAX)) 
        
        print 'init kappas...'
        self.kappas = numpy.empty((self.R_MAX,self.n), dtype=self.precision)
        # if some kappa[t, state] = -INFINITY and phi[t,state] = -INFINITY, no initialization is possilbe (e.g. not possible to choose max btw kappa and phi)
        self.kappas.fill(numpy.log(MINIMAL_PROB))
        
        for currState in range(self.n):
            sumObsProb = 0
            currRefMax = self.durationPdf.getEndDur( self.durationMap[currState])
            currLogPi = numpy.log(self.pi[currState])
            
            for t in range(1,int(currRefMax)+1):
                                
                updateQuantity, sumObsProb = self._calcUpdateQuantity(t-1, t, currState, 0, sumObsProb)
                
                self.kappas[t-1,currState] = currLogPi + updateQuantity
                
                 #sanity check. for debug
                if self.kappas[t-1,currState] == 0:
                     print "underflow error at time {}, currState {}".format(t-1, currState)
                
                
        writeListOfListToTextFile(self.kappas, None , PATH_LOGS + '/kappas') 
    
    
  
        
        
    def getMaxPhi(self, t, currState, minDur, endDur):
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
        
#         print "in getMaxPhi: maxDuration =",  currRefDur
    

        
        for d in range(minDur, endDur):

            currPhi = self.phi[t-d][fromState]
                        
            updateQuantity, sumObsProb = self._calcUpdateQuantity(t-d+1, d, currState, currPhi, sumObsProb)

            #sanity check. The '=' sign is when both are infty, take d as index
            if updateQuantity >= maxPhi:
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

    
