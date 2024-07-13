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

from classifier_imports_pristine import * 

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

def file_path_to_fingerprints_and_absolute_times(file_path: Union[str, Path], amplitude_percentile: float=0.75, fanout_number: int=15):
    """Take the music file path of a song and returns its fingerprints and absolute_times.

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
    fanout term frequency, and change in time interval, and List[int] of the abs_times of fingerprints"""

    samples = load_music_file(file_path)

    S = dig_samp_to_spec(samples)

    neighborhood = iterate_structure(generate_binary_structure(2, 1), 20)

    peak_locations = local_peak_locations(S, neighborhood, amp_min=find_cutoff_amp(S, amplitude_percentile))

    fingerprints, abs_times = local_peaks_to_fingerprints_with_absolute_times(peak_locations, fanout_number)
    return fingerprints, abs_times

