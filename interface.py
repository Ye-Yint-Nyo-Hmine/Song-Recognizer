from tkinter import *
from tkinter import filedialog
import pygame
import os

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


def add_song():
    file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
    if file_path:
        dest_path = os.path.join("music_files/mp3", os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        artist, title = os.path.basename(file_path).split(".")[0].split("--")
        song_list.append((artist, title, os.path.basename(file_path)))
        display_songs("music_files/mp3", song_list)

def recognizer():
    """
    # Todo: Implement the song recognizer here by calling the appropriate set of functions to recognize the song
    #* And then, return a list of songs from highest ranked to lowest ranked (could be song id). Lmk if you finished this
    """

def get_song_list(path_folder):
    song_list = []
    for music in os.listdir(path_folder):
        artist, title = music.split(".")[0].split("--")
        song = music
        song_list.append([artist, title, song, 0])
    return song_list

def play_song(path, song_list, index):
    global main_path
    if song_list[index][3] == 1:
        stop_song()
        song_list[index][3] = 0
        display_songs(main_path, song_list)
    else:
        for songs in song_list:
            songs[3] = 0
        song_list[index][3] = 1
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        display_songs(main_path, song_list)


def stop_song():
    print("stopping ...")
    pygame.mixer.music.stop()

def display_songs(music_path, song_list, max_show=10, x_origin=1000, y_origin=200, spacing=32):
    playlist_label = Label(text="Playlist - We Love Bytes", font=font_style_medium, bg=color_palette["bg"], 
                            fg=color_palette["accent"], borderwidth=0, relief="sunken").place(x=x_origin, y=y_origin-20)
    for index, songs in enumerate(song_list):
        artist, title, song, playing = songs
        fg = color_palette["text"]
        if playing:
            fg = color_palette["active text"]
        song_block = Button(root, text=f"{index}.  {title}  by {artist}", font=font_style_small, bg=color_palette["bg"], 
                            fg=fg, borderwidth=0, relief="sunken", 
                            activebackground=color_palette["bg"], activeforeground=color_palette["active text"], 
                            command=lambda path=os.path.join(music_path, song), song_list=song_list, index=index: play_song(path, song_list, index))
        y_origin += spacing
        song_block.place(x=x_origin, y=y_origin)



add_song_button = Button(root, text=f"Add Song", font=font_style_medium, bg=color_palette["bg"], 
                            fg=color_palette["accent"], borderwidth=0, relief="sunken", 
                            activebackground=color_palette["bg"], activeforeground=color_palette["active text"], 
                            command=add_song).place(x=30, y=500)

main_path = "music_files/mp3"
song_list = get_song_list(main_path)
display_songs(main_path, song_list)


root.mainloop()


