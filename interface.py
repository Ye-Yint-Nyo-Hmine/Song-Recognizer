from tkinter import *
from tkinter import filedialog
from classifier_imports_pristine import *
from classifier_application import *

import pygame
import os

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
import time

import uuid
import os
from pathlib import Path
from collections import Counter
import pickle
import pprint
import shutil

color_palette = {
    "bg": "#171717",
    "accent": "#d6792d",
    "text": "#ffffff",
    "active text": "#7ae043",
    "accent2": "#43e0db"
}


font_style_small = ("Helvetica", 12)
font_style_medium = ("Helvetica", 24)
font_style_large = ("Helvetica", 30, "bold")


root = Tk()
root.state('zoomed') 
root.title("We love Bytes - Music Recognizer")
root.configure(bg=color_palette["bg"])


main_label = Label(text="We Love Bytes - Song Recognizer", font=font_style_large, bg=color_palette["bg"], 
                            fg=color_palette["accent2"], borderwidth=0, relief="sunken").place(x=20, y=20)

main_path = "music_files/mp3"

pygame.mixer.init()

def add_song():
    global main_path

    file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
    if file_path:
        dest_path = os.path.join(main_path, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        song = os.path.basename(file_path) # .split(".")[0].split("--")
        song_list.append([song, 0])
        add_song_path_to_database(dest_path)
        display_songs(main_path, song_list)

def recognizer(frames):
    global main_path

    print("Sampling... ")
    samples = convert_mic_frames_to_audio(frames)

    print("Spectoring... ")
    S = dig_samp_to_spec(samples)
    
    print("Calculating Neighborhood... ")
    neighborhood = iterate_structure(generate_binary_structure(2, 1), 20)

    print("Finding Peaks... ")
    peak_locations = local_peak_locations(S, neighborhood, amp_min=find_cutoff_amp(S, 0.75))

    print("Converting Peaks to Fingerprints...")
    fingerprints_times_package = local_peaks_to_fingerprints_abs_times_match_format(peak_locations, 15)

    print("Writing to file...")
    with open("fingerprints_readable.txt", "w") as f:
        pprint.pprint(fingerprints_times_package, stream = f)

    print("Matching...")
    best_ranked_songs = match(fingerprints_times_package)
    print(f"Best ranked songs: {best_ranked_songs}")
    
    best_ranked = best_ranked_songs[0][0]

    if len(best_ranked_songs) > 1 and best_ranked_songs[0][1] == best_ranked_songs[1][1]:
        n = best_ranked_songs[0][1]
        best_ranked = []

        for song, cnt in best_ranked_songs:
            if cnt < n:
                break
            best_ranked.append(song)

    print("Best matched: ", best_ranked)
    
    if isinstance(best_ranked, list):
        best_ranked = best_ranked[0]
        print("Starting to play one best matched song...\n\n")

    else:
        print("Starting to play best matched song...\n\n")

    play_correct_song(os.path.join(main_path, best_ranked+".mp3"), get_song_list(main_path))

def play_correct_song(path, song_list):
        global main_path

        for songs in song_list:
            songs[1] = 0
        for i, (song_path, playing) in enumerate(song_list):
            if str(os.path.join(main_path, song_path)) == str(path):
                song_list[i][1] = 1
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                break

        display_songs(main_path, song_list)

def record(duration=10):
    """
    # Todo: Implement the record function here
    """
    from microphone import record_audio

    stop_song()

    listen_time = duration
    frames, sample_rate = record_audio(listen_time)
    print("IN RECORDING")
    recognizer(frames)


def get_song_list(path_folder):
    song_list = []
    for music in os.listdir(path_folder):
        # artist, title = music.split(".")[0].split("--")
        song = music
        song_list.append([song, 0])
    return song_list

def play_song(path, song_list, index):
    global main_path
    if song_list[index][1] == 1:
        stop_song()
        song_list[index][1] = 0
        display_songs(main_path, song_list)
    else:
        for songs in song_list:
            songs[1] = 0
        song_list[index][1] = 1
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        display_songs(main_path, song_list)


def get_song_duration(path):
    try:
        file_size = os.path.getsize(path)
        bitrate = 128000
        duration = int(file_size * 8 / bitrate)
        minutes = duration // 60
        seconds = duration - (60*minutes)
        return minutes, seconds
    except Exception as e:
        print(f"Error estimating MP3 file duration: {e}")
        return None


def stop_song():
    pygame.mixer.music.stop()

def display_songs(music_path, song_list, max_show=10, x_origin=50, y_origin=150, spacing=32):
    playlist_label = Label(text="Playlist - We Love Bytes", font=font_style_medium, bg=color_palette["bg"], 
                            fg=color_palette["accent"], borderwidth=0, relief="sunken").place(x=x_origin, y=y_origin-20)
    for index, songs in enumerate(song_list):
        if index >= 14 and index % 14 == 0:
            y_origin = 150
            x_origin += 550
        song, playing = songs
        fg = color_palette["text"]
        if playing:
            fg = color_palette["active text"]
        song_duration = get_song_duration(os.path.join(music_path, song))
        song_block = Button(root, text=f"{song_duration[0]:02}:{song_duration[1]:02}.  {song}", font=font_style_small, bg=color_palette["bg"],
                            fg=fg, borderwidth=0, relief="sunken", 
                            activebackground=color_palette["bg"], activeforeground=color_palette["active text"], 
                            command=lambda path=os.path.join(music_path, song), song_list=song_list, index=index: play_song(path, song_list, index))
        y_origin += spacing
        song_block.place(x=x_origin, y=y_origin)


        

main_buttons_y = 750
add_song_button = Button(root, text=f"Add Song", font=font_style_medium, bg=color_palette["bg"], 
                            fg=color_palette["accent"], borderwidth=0, relief="sunken", 
                            activebackground=color_palette["bg"], activeforeground=color_palette["active text"], 
                            command=add_song).place(x=50, y=main_buttons_y)
record_song_button = Button(root, text=f"Recognize", font=font_style_medium, bg=color_palette["bg"], 
                            fg=color_palette["active text"], borderwidth=0, relief="sunken", 
                            activebackground=color_palette["bg"], activeforeground=color_palette["accent2"], 
                            command=record).place(x=300, y=main_buttons_y)



song_list = get_song_list(main_path)
display_songs(main_path, song_list)
load_initial_database()

root.mainloop()
