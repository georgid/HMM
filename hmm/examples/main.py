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
from hmm.Parameters import Parameters, DEVIATION_IN_SEC
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
    lyricsWithModels = LyricsWithModels(lyrics, htkParser, 'False', DEVIATION_IN_SEC)
     
    observationFeatures = loadMFCCs(URIrecordingNoExt, withSynthesis, fromTs, toTs) #     observationFeatures = observationFeatures[0:1000]

     
    lyricsWithModels.duration2numFrameDuration(observationFeatures, URIrecordingNoExt)
#     lyricsWithModels.printWordsAndStates()
    
    return lyricsWithModels, observationFeatures



def decode(lyricsWithModels, observationFeatures):   
    '''
    same as decoder.decodeAudio() without the parts with WITH_Duration flag.
    '''
    alpha = 0.97
    ONLY_MIDDLE_STATE=False
    params = Parameters(alpha, ONLY_MIDDLE_STATE)
    decoder = Decoder(lyricsWithModels, params.ALPHA)
    
    
    #  decodes
    decoder.hmmNetwork.initDecodingParameters(observationFeatures)
    chiBackPointer, psiBackPointer = decoder.hmmNetwork._viterbiForcedDur(observationFeatures)
    
    backtrack(chiBackPointer, psiBackPointer, decoder)


def backtrack(chiBackPointer,psiBackPointer, decoder ):
    # backtrack
    path =  Path(chiBackPointer, psiBackPointer)
    detectedWordList = decoder.path2ResultWordList(path)
         # DEBUG
    
    decoder.lyricsWithModels.printWordsAndStatesAndDurations(decoder.path)
    path.printDurations()
    

    
# if __name__ == '__main__':    
# 
#     
#     withSynthesis = False
#     lyrics = loadLyrics(pathToComposition, whichSection, withSynthesis)
#     lyricsWithModels, observationFeatures = loadSmallAudioFragment(lyrics,  URIrecordingNoExt, withSynthesis, fromTs=-1, toTs=-1)
#      
#     decode(lyricsWithModels, observationFeatures)