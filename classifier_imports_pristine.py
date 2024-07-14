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

import uuid
import os
from pathlib import Path
from collections import Counter
import pickle

SAMPLING_RATE = 44100

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

    SAMPLING_RATE = 44100
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
dict_data_to_id = {}
def process_all_songs(directory_path: str, num_fanout: int = 5):
    global dict_data_to_id
    
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")

            # Load and process the audio file
            audio = load_music_file(file_path)
            S = dig_samp_to_spec(audio)

            # Define neighborhood structure for peak detection
            neighborhood = generate_binary_structure(2, 1)
            neighborhood = iterate_structure(neighborhood, 20)

            # Detect peaks
            amp_min = find_cutoff_amp(S, 0.77)
            peaks = local_peak_locations(S, neighborhood, amp_min)

            # Generate fingerprints
            fingerprints = local_peaks_to_fingerprints(peaks, num_fanout)
            song_id = str(uuid.uuid4())

            # Store fingerprints in the dictionary
            for fp in fingerprints:
                if fp not in dict_data_to_id:
                    dict_data_to_id[fp] = []
                dict_data_to_id[fp].append((song_id, fp[2]))

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
    """Returns the fingerprint a set of peaks packaged as a tuple.

    Parameters
    ----------
    local_peaks : List[Tuple[int, int]]
        List of row, column (frequency, time) indexes of the peaks

    num_fanout : int
         Number of fanout points for each reference point

    Returns
    -------
    List[Tuple[int, int, int]]
        List of fingerprints"""
    
    result = [] #should be a list of lists

    if num_fanout <= len(local_peaks):
        for i in range(len(local_peaks) - num_fanout): # subtract because it had to be only peaks after, and dont want index out of bounds error
            i_fingerprints = []
            i_freq, i_time = local_peaks[i]
            for j in range(1, num_fanout+1):
                f_freq, f_time = local_peaks[i+j]
                i_fingerprints.append((i_freq, f_freq, f_time - i_time))
            
            result += i_fingerprints # contatenate lists
        
        return result # should be a 2d list, that can then be zipped w the peaks if we need to know which peak its associated with
    else:
        return "IndexError"

def local_peaks_to_fingerprints_with_absolute_times(local_peaks: List[Tuple[int, int]], num_fanout: int):
    """Returns the fingerprint and absolute time of the fingerprint of a set of peaks.

    Parameters
    ----------
    local_peaks : List[Tuple[int, int]]
        List of row, column (frequency, time) indexes of the peaks

    num_fanout : int
         Number of fanout points for each reference point

    Returns
    -------
    List[Tuple[int, int, int]] contained the reference point frequency, 
    fanout term frequency, and change in time interval, and List[int] of the abs_times of fingerprints."""

    fingerprints = []
    abs_times = []
    
    if num_fanout <= len(local_peaks):
        for i in range(len(local_peaks) - num_fanout): # subtract because it had to be only peaks after, and dont want index out of bounds error
            i_freq, i_time = local_peaks[i]
            for j in range(1, num_fanout+1):
                f_freq, f_time = local_peaks[i+j]
                fingerprints.append((i_freq, f_freq, f_time - i_time))
                abs_times.append(i_time)
            
        
        return fingerprints, abs_times
    else:
        return "IndexError"
    
def process_all_songs(directory_path: str, num_fanout: int = 5):
    global dict_data_to_id
    dict_data_to_id = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")

            # Load and process the audio file
            audio = load_music_file(file_path)
            S = dig_samp_to_spec(audio)

            # Define neighborhood structure for peak detection
            neighborhood = generate_binary_structure(2, 1)
            neighborhood = iterate_structure(neighborhood, 20)

            # Detect peaks
            amp_min = find_cutoff_amp(S, 0.77)
            peaks = local_peak_locations(S, neighborhood, amp_min)

            # Generate fingerprints
            fingerprints = local_peaks_to_fingerprints(peaks, num_fanout)
            song_id = str(uuid.uuid4())

            # Store fingerprints in the dictionary
            for fp in fingerprints:
                if fp not in dict_data_to_id:
                    dict_data_to_id[fp] = []
                dict_data_to_id[fp].append((song_id, fp[2]))

    print(f"Processed {len(dict_data_to_id)} unique fingerprints from {len(os.listdir(directory_path))} songs.")
    return dict_data_to_id
    
    
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

def get_accuracy_test(song_fingerprints: Tuple[int, int, int], test_fingerprints: Tuple[int, int, int]):
    """Returns the percent match as a decimal of a test clip compared to a pristine studio recording."""

    matches = [fp for fp in test_fingerprints if fp in song_fingerprints]

    return len(matches) / len(test_fingerprints)

def local_peaks_to_fingerprints_abs_times_match_format(local_peaks: List[Tuple[int, int]], num_fanout: int):
    """Returns the fingerprint and absolute time of the fingerprint of a set of peaks.

    Parameters
    ----------
    local_peaks : List[Tuple[int, int]]
        List of row, column (frequency, time) indexes of the peaks

    num_fanout : int
         Number of fanout points for each reference point

    Returns
    -------
    List[Tuple[Tuple[int, int, int], int]]
        List of fingerprints with the absolute time of the fingerprint tupled together"""

    result = [] #should be a list of tuples

    if num_fanout <= len(local_peaks):
        for i in range(len(local_peaks) - num_fanout): # subtract because it had to be only peaks after, and dont want index out of bounds error
            i_fingerprints_times = []
            i_freq, i_time = local_peaks[i]
            for j in range(1, num_fanout+1):
                f_freq, f_time = local_peaks[i+j]
                i_fingerprints_times.append(((i_freq, f_freq, f_time - i_time), i_time))
            
            result += i_fingerprints_times # contatenate lists
        
        return result
    else:
        return "IndexError"
    
def get_songs_with_fp(fingerprint: Tuple[Tuple[int, int, int], int]):
    """
    Traverses database for matching input-fingerprint
    Returns list of songs [(song ID,offset),...] in which input-fingerprint occurs
    """
    
    """songs=[]
    
    for fp in dict_data_to_id.keys():
        
        if fingerprint[0] == fp[0]:
            #appends ( ('song ID', absolute time when of peak), offset)
            for matching_fp in dict_data_to_id[fp]:
                songs.append( (matching_fp[0], matching_fp[1] - fingerprint[1]) )

    #print(songs)
    return songs"""

    print("match function successful")
    dict_data_to_id, dict_id_to_song = {}, {}
    with open('fingerprint_database.pkl', 'rb') as f:
        dict_data_to_id = pickle.load(f)
    with open('id_to_song_dictionary.pkl', 'rb') as f:
        dict_id_to_song = pickle.load(f)
    
    song_ids_with_abs_times = dict_data_to_id[fingerprint[0]]
    songs_offsets = []
    for id, abs_t in song_ids_with_abs_times:
        songs_offsets.append((dict_id_to_song[id],  abs_t - fingerprint[1]))

    return songs_offsets

def match(test_fingerprints):
    from collections import Counter
    """
    Traverses input-fingerprints
    Finds songs with fingerprints
    Returns song with most occurances of test_fingerprints
    """
    print("IN MATCH")
    dict_data_to_id, dict_id_to_song = {}, {}
    with open('fingerprint_database.pkl', 'rb') as f:
        dict_data_to_id = pickle.load(f)
    with open('id_to_song_dictionary.pkl', 'rb') as f:
        dict_id_to_song = pickle.load(f)
        # print(dict_id_to_song)
    
    songs_offsets=[]
    songs = []

    for fp in test_fingerprints:
        #print(fp in dict_data_to_id)
        if fp[0] in dict_data_to_id:
            song_and_offset = get_songs_with_fp(fp)
            songs_offsets += song_and_offset
            songs += song_and_offset[0]

    # counted_songs = Counter(songs).most_common().sort()            
            
    return Counter(songs_offsets).most_common() # [0] # [0][0] #remove indexes if error, returns song



if __name__ == '__main__':
    frames = load_music_file("music_files\mp3\Elliott-Smith--Pitseleh.mp3")
    print("Sampling... ")
    samples = convert_mic_frames_to_audio(frames)
    
    
    print("Processing all songs...")
    # process_all_songs("music_files\mp3")
    
    print("Spectoring... ")
    S = dig_samp_to_spec(samples)
    
    print("Calculating Neighborhood... ")
    neighborhood = iterate_structure(generate_binary_structure(2, 1), 20)

    print("Finding Peaks... ")
    peak_locations = local_peak_locations(S, neighborhood, amp_min=find_cutoff_amp(S, 0.75))

    print("Converting Peaks to Fingerprints ...")
    fingerprints_times_package = local_peaks_to_fingerprints_abs_times_match_format(peak_locations, 15)
    print("Matching... ")
    best_ranked_song_offsets = match(fingerprints_times_package) # this should be changed to the best ranked matched song PATH***
    print(f"Best ranked song-offset pairs: {best_ranked_song_offsets}")
    
    best_ranked = best_ranked_song_offsets[0][0]

    if len(best_ranked_song_offsets) > 1 and best_ranked_song_offsets[0][1] == best_ranked_song_offsets[1][1]:
        songs = [song_offset[0] for song_offset in best_ranked_song_offsets]
        songs_counted = Counter(songs)
        best_ranked = songs_counted.most_common(1)[0]

    print("Best matched: ", best_ranked)
