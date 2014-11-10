'''
Created on Oct 31, 2014

@author: joro
'''
import numpy
import os
import sys

from numpy.core.numeric import Infinity
from numpy.distutils.core import numpy_cmdclass

from Utilz import writeListOfListToTextFile
from _ContinuousHMM import _ContinuousHMM
from hmm.continuous.DurationPdf  import DurationPdf


parentDir = os.path.abspath(  os.path.join(os.path.dirname(os.path.realpath(sys.argv[0]) ), os.path.pardir,  os.path.pardir ) ) 
pathUtils = os.path.join(parentDir, 'utilsLyrics')
if pathUtils not in sys.path: sys.path.append(pathUtils )


# TODO: take max from durations


class _DurationHMM(_ContinuousHMM):
    '''
    Implements the decoding with duration probabilities, but should not be used directly.
    '''
    
    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=numpy.double,verbose=False):
            '''
            See _ContinuousHMM constructor for more information
            '''
            _ContinuousHMM.__init__(self,n,m,d,A,means,covars,w,pi,min_std,init_type,precision,verbose) #@UndefinedVariable
            
                      
    
    def setDurForStates(self, listDurations):
        '''
        mapping of state to its duration (in number of frames).
        @param listDurations read from  state.Durations
        
        '''
        if len(listDurations) != self.n:
            sys.exit("not exact duration")
            
        self.durationMap =  numpy.array(listDurations, dtype=int)

#         STUB
#         self.durationMap =  numpy.arange(1,self.n+1)
#         
       
        # set duration lookup table
        self.MAX_DUR = int(numpy.amax(self.durationMap))
        self.durationPdf = DurationPdf(self.MAX_DUR)

    def getWaitLogLik(self, d, state):
        '''
        return waiting pdf. uses gamma
        IMPORTANT: if d>D function should still return values up to some limit (e.g. +100% and least till MaxDur)
        STUB
        '''  
        
        DMaxCurrState = self.durationMap[state]
        
        return self.durationPdf.getWaitLogLik(d, DMaxCurrState)
         
    
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
        
        writeListOfListToTextFile(self.phi, None , '/tmp/phi') 
            
        # return for backtracking
        return  self.chi, self.psi
    
    
    def _calcCurrStatePhi(self,  t, currState):
        '''
        calc. quantities in recursion  equation
        '''
        maxD_currState = self.durationMap[currState]
        
        #TODO take 20% more from dur from score 
        maxPhi, fromState,  maxDurIndex =  self.getMaxPhi(t, currState, maxD_currState)
                
        self.phi[t][currState] = maxPhi
        
        self.psi[t][currState] = fromState
        
        self.chi[t][currState] = maxDurIndex
                
           
    def _initBeginingPhis(self, lenObservations):
        '''
        initialization function
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
            for currState in range(1, self.n): 
                phiStar, fromState,  maxDurIndex  =  self.getMaxPhi(t, currState, t)
                
                if  phiStar > self.kappas[t,currState] :
                    self.phi[t,currState] = phiStar
                    self.psi[t,currState] = fromState
                    self.chi[t,currState] = maxDurIndex
                else:
                    self.phi[t, currState] = self.kappas[t, currState]
                    # kappas mean still at beginning state
                    self.psi[t,currState] = currState
                    self.chi[t,currState] = t
                    
        
        writeListOfListToTextFile(self.phi, None , '/tmp/phi_init') 

        
    def _initKappas(self):
        '''
        kappas[t][s] - starting and staying at time t in same currState s.
        WITH LogLik 
        '''
        print 'init kappas...'
        self.kappas = numpy.empty((self.MAX_DUR,self.n), dtype=self.precision)
        self.kappas.fill(-Infinity)
        
        for currState in range(self.n):
            sum = numpy.log(self.pi[currState]) 
            D = self.durationMap[currState]
            
            for d in range(1,int(D)+1):
                sum += self.B_map[currState, d-1]
                quant = (sum + self.getWaitLogLik(d, currState))
                self.kappas[d-1,currState] = quant
                
                 #sanity check
                if self.kappas[d-1,currState] == 0:
                     print "underflow error at time {}, currState {}".format(d-1, currState)
                
                
        writeListOfListToTextFile(self.kappas, None , '/tmp/kappas') 
    
    
  
        
        
    def getMaxPhi(self, t, currState, maxPossibleDuration):
        '''
        recursive rule. Find duration that maximizes current phi
        @return: maxPhi - pprob
        @return: fromState - from which state we come (hard coded to prev. state in forced alignment) 
        @return: maxDurIndex - index Duration with max prob. INdex for t begins at 0
        '''
        sumObsProb = 0
         # due to forced alignment
        fromState = currState - 1
        maxPhi = -1 * numpy.Infinity 
        maxDurIndex = -1
          
        for d in range(1, maxPossibleDuration+1):
            sumObsProb += self.B_map[currState, t-d+1]
            currPhi = self.phi[t-d][fromState]
            updateQuantity = currPhi + self.getWaitLogLik(d, currState) + sumObsProb
            #sanity check
            # DEBUG:
            if updateQuantity == -Infinity:
                print "some prob=0 at time {}, state {}, duration {}".format(t, currState, d)
            
            if updateQuantity > maxPhi:
                maxPhi = updateQuantity
                maxDurIndex = d
        
        return maxPhi, fromState, maxDurIndex    
    
   
             
