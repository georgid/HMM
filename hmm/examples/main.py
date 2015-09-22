# -*- coding: utf-8 -*-

'''
Created on Nov 13, 2012

@author: GuyZ
'''

import numpy

from hmm.continuous.GMHMM import GMHMM
from hmm.discrete.DiscreteHMM import DiscreteHMM
import os
import sys
from hmm.Parameters import Parameters
from hmm.continuous.DurationPdf import NUMFRAMESPERSEC

# parentParentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__) ), os.path.pardir, os.path.pardir, os.path.pardir)) 
# pathJingju = os.path.join(parentParentDir, 'Jingju')
# 
# if pathJingju not in sys.path:
#     sys.path.append(pathJingju )
from hmm.Path import Path
from hmm.ParametersAlgo import ParametersAlgo
import logging


# file parsing tools as external lib 
parentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__) ), os.path.pardir, os.path.pardir,os.path.pardir )) 
print parentDir

pathJingjuAlignment = os.path.join(parentDir, 'AlignmentDuration')
if not pathJingjuAlignment in sys.path:
    sys.path.append(pathJingjuAlignment)
    
from Phonetizer import Phonetizer
from MakamScore import loadLyrics
from LyricsWithModels import LyricsWithModels
from Decoder import Decoder
from FeatureExtractor import loadMFCCs

modelDIR = pathJingjuAlignment + '/model/'
HMM_LIST_URI = modelDIR + '/monophones0'
MODEL_URI = modelDIR + '/hmmdefs9gmm9iter'

# parser of htk-build speech model
pathHtkModelParser = os.path.join(parentDir, 'pathHtkModelParser')
sys.path.append(pathHtkModelParser)
from htk_converter import HtkConverter





def loadSmallAudioFragment(lyrics, URIrecordingNoExt, withSynthesis, fromTs=-1, toTs=-1):
    '''
    test duration-explicit HMM with audio features from real recording and htk-loaded model
    asserts it works. no results provided 
    '''
    
    print fromTs
    htkParser = HtkConverter()
    htkParser.load(MODEL_URI, HMM_LIST_URI)
    lyricsWithModels = LyricsWithModels(lyrics, htkParser, 'False', ParametersAlgo.DEVIATION_IN_SEC)
     
    observationFeatures, URIRecordingChunk = loadMFCCs(URIrecordingNoExt, withSynthesis, fromTs, toTs) #     observationFeatures = observationFeatures[0:1000]


    lyricsWithModels.duration2numFrameDuration(observationFeatures, URIrecordingNoExt)
#     lyricsWithModels.printPhonemeNetwork()


    
    return lyricsWithModels, observationFeatures, URIRecordingChunk


def decodeWithOracle(lyrics, URIrecordingNoExt, fromTs, toTs, fromPhonemeIdx, toPhonemeIdx):
    '''
    instead of bMap  set as oracle from annotation
    '''
    ParametersAlgo.DEVIATION_IN_SEC = 0.1
    # synthesis not needed really in this setting. workaround because without synth takes whole recording  
    withSynthesis = 1
    htkParser = HtkConverter()
    htkParser.load(MODEL_URI, HMM_LIST_URI)
    
    dummyDeviation = 1
    # lyricsWithModelsORacle used only as helper for state durs, but not functionally
    lyricsWithModelsORacle = LyricsWithModels(lyrics, htkParser, 'False', dummyDeviation)
    lyricsWithModelsORacle.setPhonemeDurs( URIrecordingNoExt + '.TextGrid', fromPhonemeIdx, toPhonemeIdx)
    
#     lyricsWithModels, observationFeatures = loadSmallAudioFragment(lyrics,  URIrecordingNoExt, withSynthesis, fromTs=-1, toTs=-1)
    lyricsWithModels, observationFeatures, URIRecordingChunk = loadSmallAudioFragment(lyrics,  URIrecordingNoExt, withSynthesis, fromTs, toTs)
    

    decoder = getDecoder(lyricsWithModels, URIRecordingChunk)
    
    lenObservations = decoder.hmmNetwork.initDecodingParametersOracle(lyricsWithModelsORacle, URIrecordingNoExt, fromTs, toTs)
    
    chiBackPointer, psiBackPointer = decoder.hmmNetwork._viterbiForcedDur(lenObservations)
#     
    detectedWordList, path = decoder.backtrack(chiBackPointer, psiBackPointer)
    return detectedWordList



def getDecoder(lyricsWithModels, URIrecordingNoExt):
    '''
    helper routine to init decoder. change here parameters
    '''
    alpha = 0.97
    ONLY_MIDDLE_STATE=False
    params = Parameters(alpha, ONLY_MIDDLE_STATE)
    decoder = Decoder(lyricsWithModels, URIrecordingNoExt, params.ALPHA)
    return decoder


def decode(lyricsWithModels, observationFeatures, URIrecordingNoExt):   
    '''
    convenience method. same as decoder.decodeAudio() without the parts with WITH_Duration flag.
    '''
    decoder = getDecoder(lyricsWithModels, URIrecordingNoExt)
    
    #  decodes
    decoder.hmmNetwork.initDecodingParameters(observationFeatures)
    lenObs = len(observationFeatures)
    chiBackPointer, psiBackPointer = decoder.hmmNetwork._viterbiForcedDur(lenObs)
#     
    decoder.backtrack(chiBackPointer, psiBackPointer)



    
# if __name__ == '__main__':    
# 
#     
#     withSynthesis = False
#     lyrics = loadLyrics(pathToComposition, whichSection, withSynthesis)
#     lyricsWithModels, observationFeatures, URIrecordingChunk = loadSmallAudioFragment(lyrics,  URIrecordingNoExt, withSynthesis, fromTs=-1, toTs=-1)
#      
#     decode(lyricsWithModels, observationFeatures)