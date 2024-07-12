### Dependencies ###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
# from microphone import record_audio # add if utilizing microphone and in Microphone directory
from IPython.display import Audio
from typing import Tuple
import librosa

from numba import njit
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import iterate_structure

from typing import Tuple, Callable, List, Union

import os
from pathlib import Path

### Load audio file from database ###

def load_music_file(file_path: str):
    """Loads a target music file path.

    Parameters
    ----------
    file_path : str
        File path of song
        
    Returns
    -------
    recorded_audio: np.ndarray
        Audio samples
    """

    audio, samp_rate = librosa.load(file_path, sr=SAMPLING_RATE, mono=True)
    return audio

### Mic audio to integers ###

def convert_mic_frames_to_audio(frames: np.ndarray) -> np.ndarray:
    """Converts frames taken from microphone to 16-bit integers
    
    Parameters
    ----------
    frames : np.ndarray
        List of bytes recorded from a microphone
        
    Returns
    -------
    numpy.ndarray
        Bytes converted to 16-bit integers
    """

    return np.hstack([np.frombuffer(i, np.int16) for i in frames])

### Clip Maker function to test classifier ###

def make_random_clips(samples: np.ndarray, *, desired_length: int, count: int):
    """Takes audio samples and cuts {count} number of {desired_length} clips.

    Parameters
    ----------
    samples: np.ndarray
        Array of audio samples

    desired_length: int
        Length of each clip in seconds

    count: int
        Total number of clips
        
    Returns
    -------
    np.ndarray, shape-(count,N)
        2-D array with {count} number of clip samples
    """
    import random
    
    N = len(samples)
    sampling_rate = SAMPLING_RATE
    T = N / sampling_rate
    assert desired_length < T, "length of clip cannot be greater than song length"
    
    percent_of_duration = desired_length / T
    samples_per_clip = int(percent_of_duration * len(samples))
    
    clip_samples = []

    for i in range(count):
        random_sample_idx = random.randrange(0, N - samples_per_clip)
        clip_sample = samples[random_sample_idx : random_sample_idx + samples_per_clip]
        clip_samples.append(clip_sample)

    return np.array(clip_samples)

### Plot and return figure and axes for Spectrogram ###

def dig_samp_to_spec_plot (samples: np.ndarray):
    # data = np.hstack([np.frombuffer(i, np.int16) for i in frames])

    # using matplotlib's built-in spectrogram function
    fig, ax = plt.subplots()

    S, freqs, times, im = ax.specgram(
        samples,
        NFFT=4096,
        Fs=SAMPLING_RATE,
        window=mlab.window_hanning,
        noverlap=4096 // 2,
        mode='magnitude'
    )
    ax.set_ylim(0, 10000)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("frequency (Hz)")
    return fig, ax

### Return Spectrogram 2-D Array ###

def dig_samp_to_spec(samples: np.ndarray):
    """Takes a 1-D sampled audio array and returns a 2-D spectrogram."""
    
    S, freqs, times = mlab.specgram(
    samples,
    NFFT=4096,
    Fs=SAMPLING_RATE,
    window=mlab.window_hanning,
    noverlap=int(4096 / 2),
    mode='magnitude'
    )

    return S

### Return the peaks of a spectrogram ###

@njit
def _peaks(data_2d, rows, cols, amp_min):
    """
    A Numba-optimized 2-D peak-finding algorithm.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    rows : numpy.ndarray, shape-(N,)
        The 0-centered row indices of the local neighborhood mask
    
    cols : numpy.ndarray, shape-(N,)
        The 0-centered column indices of the local neighborhood mask
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location. 
    """
    peaks = []
    
    # iterate over the 2-D data in col-major order
    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            continue

        for dr, dc in zip(rows, cols):
            if dr == 0 and dc == 0:
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                dr *= -1

            if not (0 <= c + dc < data_2d.shape[1]):
                dc *= -1

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                break
        else:
            peaks.append((r, c))
    return peaks

def local_peak_locations(data_2d, neighborhood, amp_min):
    """
    From 
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min`.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected
    
    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location.
    
    Notes
    -----
    The local peaks are returned in column-major order.
    """
    rows, cols = np.where(neighborhood)
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1

    rows -= neighborhood.shape[0] // 2
    cols -= neighborhood.shape[1] // 2
    
    return _peaks(data_2d, rows, cols, amp_min=amp_min)

### Turn peaks to fingerprints ###

def local_peaks_to_fingerprints(local_peaks: List[Tuple[int, int]], num_fanout: int):

    result = [] #should be a list of lists

    if num_fanout <= len(local_peaks):
        for i in range(len(local_peaks) - num_fanout): # subtract because it had to be only peaks after, and dont want index out of bounds error
            i_fingerprints = []
            i_freq, i_time = local_peaks[i]
            for j in range(1, num_fanout+1):
                f_freq, f_time = local_peaks[i+j]
                i_fingerprints += [(i_freq, f_freq, f_time - i_time)]
            
            result += i_fingerprints
        
        return result # should be a 2d list, that can then be zipped w the peaks if we need to know which peak its associated with
    else:
        return "IndexError"
    
### Set cutoff for amplitude to disregard background noise in real world samples ###

def find_cutoff_amp(S: np.ndarray, percentile: float):
    """Returns the log_amplitude of a target spectrogram that will be the cutoff for background noise
       in real world samples. Calculated using decimal part percentile.

    Parameters
    ----------
    S : numpy.ndarray
        The target spectrogram

    percentile : float
         A demical < 1.0 for which the cutoff is greater than or equal to the {percentile}
         percentile of log_amplitudes

    Returns
    -------
    Cutoff amplitude"""

    S = S.ravel()  # ravel flattens 2D spectrogram into a 1D array
    ind = round(len(S) * percentile)  # find the index associated with the percentile amplitude
    cutoff_amplitude = np.partition(S, ind)[ind]  # find the actual percentile amplitude
    
    return cutoff_amplitude
