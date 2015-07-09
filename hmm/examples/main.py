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
from fpformat import decoder
from hmm.continuous.DurationPdf import NUMFRAMESPERSEC

parentParentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__) ), os.path.pardir, os.path.pardir, os.path.pardir)) 
pathJingju = os.path.join(parentParentDir, 'Jingju')

if pathJingju not in sys.path:
    sys.path.append(pathJingju )
from ParametersAlgo import ParametersAlgo
from hmm.Path import Path


# file parsing tools as external lib 
parentDir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__) ), os.path.pardir, os.path.pardir,os.path.pardir )) 
print parentDir

pathAlignmentDuration = os.path.join(parentDir, 'AlignmentDuration')
if not pathAlignmentDuration in sys.path:
    sys.path.append(pathAlignmentDuration)
    
from Phonetizer import Phonetizer
from MakamScore import loadLyrics
from LyricsWithModels import LyricsWithModels
from Decoder import Decoder
from FeatureExtractor import loadMFCCs

modelDIR = pathAlignmentDuration + '/model/'
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
 
    htkParser = HtkConverter()
    htkParser.load(MODEL_URI, HMM_LIST_URI)
    lyricsWithModels = LyricsWithModels(lyrics, htkParser, 'False', ParametersAlgo.DEVIATION_IN_SEC)
     
    observationFeatures, URIRecordingChunk = loadMFCCs(URIrecordingNoExt, withSynthesis, fromTs, toTs) #     observationFeatures = observationFeatures[0:1000]

     
    lyricsWithModels.duration2numFrameDuration(observationFeatures, URIrecordingNoExt)
#     lyricsWithModels.printPhonemeNetwork()

#     lyricsWithModels.printWordsAndStates()
    
    return lyricsWithModels, observationFeatures


def decodeWithOracle(lyrics, URIrecordingNoExt, fromTs, toTs, fromPhonemeIdx, toPhonemeIdx):
    '''
    instead map set as oracle from annotation
    '''
    
    withSynthesis = 0
    htkParser = HtkConverter()
    htkParser.load(MODEL_URI, HMM_LIST_URI)
    lyricsWithModelsORacle = LyricsWithModels(lyrics, htkParser, 'False', ParametersAlgo.DEVIATION_IN_SEC)
                                        
    lyricsWithModelsORacle.setPhonemeDurs( URIrecordingNoExt + '.TextGrid', fromPhonemeIdx, toPhonemeIdx)
#     lyricsWithModelsORacle.printPhonemeNetwork()
    
    lyricsWithModels, observationFeatures = loadSmallAudioFragment(lyrics,  URIrecordingNoExt, withSynthesis, fromTs=-1, toTs=-1)

    decoder = getDecoder(lyricsWithModels)
    
    lenObservations = decoder.hmmNetwork.initDecodingParametersOracle(lyricsWithModelsORacle, URIrecordingNoExt, fromTs, toTs)
    
    chiBackPointer, psiBackPointer = decoder.hmmNetwork._viterbiForcedDur(lenObservations)
#     
    detectedWordList = backtrack(chiBackPointer, psiBackPointer, decoder)
    return detectedWordList

def getDecoder(lyricsWithModels):
    '''
    helper routine to init decoder. change here parameters
    '''
    alpha = 0.97
    ONLY_MIDDLE_STATE=False
    params = Parameters(alpha, ONLY_MIDDLE_STATE)
    decoder = Decoder(lyricsWithModels, params.ALPHA)
    return decoder


def decode(lyricsWithModels, observationFeatures):   
    '''
    same as decoder.decodeAudio() without the parts with WITH_Duration flag.
    '''
    decoder = getDecoder(lyricsWithModels)
    
    #  decodes
    decoder.hmmNetwork.initDecodingParameters(observationFeatures)
    lenObs = len(observationFeatures)
    chiBackPointer, psiBackPointer = decoder.hmmNetwork._viterbiForcedDur(lenObs)
#     
    backtrack(chiBackPointer, psiBackPointer, decoder)


def backtrack(chiBackPointer,psiBackPointer, decoder ):
    # backtrack
    path =  Path(chiBackPointer, psiBackPointer)
    detectedWordList = decoder.path2ResultWordList(path)
         # DEBUG
    
    decoder.lyricsWithModels.printWordsAndStatesAndDurations(decoder.path)
    path.printDurations()
    return detectedWordList
    
# if __name__ == '__main__':    
# 
#     
#     withSynthesis = False
#     lyrics = loadLyrics(pathToComposition, whichSection, withSynthesis)
#     lyricsWithModels, observationFeatures = loadSmallAudioFragment(lyrics,  URIrecordingNoExt, withSynthesis, fromTs=-1, toTs=-1)
#      
#     decode(lyricsWithModels, observationFeatures)