### Dependencies ### 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from microphone import record_audio # add if utilizing microphone and in Microphone directory
from IPython.display import Audio
from typing import Tuple
import librosa
import operator

from numba import njit
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import iterate_structure

from typing import Tuple, Callable, List, Union

import uuid
import os
from pathlib import Path
from collections import Counter

from classifier_imports import * 

def file_path_to_fingerprints(file_path: Union[str, Path], amplitude_percentile: float=0.75, fanout_number: int=15):
    """Take the music file path of a song and returns it's fingerprints.

    Parameters
    ----------
    file_path : Union[str, Path]
        File path for music file

    amplitude_percentile : float, optional
         A demical < 1.0 for which all amplitudes less than the {percentile}
         percentile of amplitudes will be disregarded

    fanout_number: int, optional
        Number of fanouts for each reference point/peak in the spectrogram

    Returns
    -------
    List[Tuple[int, int, int]] contained the reference point frequency, 
    fanout term frequency, and change in time interval"""

    samples = load_music_file(file_path)

    S = dig_samp_to_spec(samples)

    neighborhood = iterate_structure(generate_binary_structure(2, 1), 20)

    peak_locations = local_peak_locations(S, neighborhood, amp_min=find_cutoff_amp(S, amplitude_percentile))

    fingerprints = local_peaks_to_fingerprints(peak_locations, fanout_number)
    return fingerprints

def get_accuracy_test(song_fingerprints: Tuple[int, int, int], test_fingerprints: Tuple[int, int, int]):
    """Returns the percent match as a decimal of a test clip compared to a pristine studio recording."""

    matches = [fp for fp in test_fingerprints if fp in song_fingerprints]

    return len(matches) / len(test_fingerprints)

def get_songs_with_fp(fingerprint: Tuple[Tuple[int, int, int], int]):
    """
    Traverses database for matching input-fingerprint
    Returns list of songs [(song ID,offset),...] in which input-fingerprint occurs
    """
    songs=[]
    for fp in dict_data_to_id.keys():
        if fingerprint[0] in fp[0]:
            songs += (dict_data_to_id[fp][0],dict_data_to_id[fp][1]-fingerprint[1])
    return songs

#check if this works with the offput in tuple
def match(test_fingerprints):
    """
    Traverses input-fingerprints
    Finds songs with fingerprints
    Returns song with most occurances of test_fingerprints
    """
    songs=[]

    for fingerprint in test_fingerprints:
        songs.append(get_songs_with_fp(fingerprint))

    most_common_song = Counter(songs).most_common()
