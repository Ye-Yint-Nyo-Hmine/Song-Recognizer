# Song Recognizer

"""
> Theses are the todo things from the documents
Creating functions for converting all variety of audio recordings, 
be them recorded from the microphone or digital audio files, into a NumPy-array of digital samples.

Creating a function that takes in digital samples of a song/recording and produces a spectrogram of log-scaled amplitudes 
and extracts local peaks from it

Creating a function that takes the peaks from the spectrogram and forms fingerprints via “fanout” patterns among the peaks.

Devising a scheme for organizing song metadata, e.g. associating song titles and artist names with a recording, 
and associating these with unique song-IDs to be used within the database.

Writing the core functionality for storing fingerprints in the database, 
as well as querying the database and tallying the results of the query.

Designing an interface for the database,
including the following functionality:

saving and loading the database

inspecting the list of songs (and perhaps artists) that exist in the database

providing the ability to switch databases (optional)

deleting a song from a database (optional)

guarding against the adding the same song to the database multiple times (optional)

Recording long clips of songs under various noise conditions (e.g. some should be clips from studio recordings, 
others recorded with little background noise, some with moderate background noise, etc.) so that you can begin to test and analyze the performance of your algorithm.

Creating a function that can take an array of audio samples from a long (e.g. one minute) recording 
and produce random clips of it at a desired, shorter length. This can help with experimentation/analysis. 
For example you can record a 1 minutes clip of a song, played from your phone and then create many random 10 second clips from it 
and see if they all successfully match against your database.
    
"""

"""
# TODO: 
1. Plan the structure for the code (ie: functions we need)
2. Create a strategy to assign tasks different people
"""

"""
These are the functions I think we should create so far (feel free to change/add on):

- song_to_sample
- record_to_sample
- PCM
- pressure
- fft_complex_to_real
- find_peaks
- fingerprints

"""