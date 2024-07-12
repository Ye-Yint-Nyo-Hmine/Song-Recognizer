# Better Shazam and Spotify: Song Recognizer Cog*Work Week 1
## Brought to you by We_Love_Bytes
![image](https://github.com/user-attachments/assets/effc65b3-6066-4bcd-8092-b9cadbfa9213)
<br/>
<br/>
#### Ever hear a song but can’t remember the name? Well download our Python Script.

### Installing environment
To start with an environment, I suggest using Conda

[WINDOWS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) 

[MAC](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)


### Creating environment
To create an environment, in your terminal, replace the following ```env_name``` with the preferred name:

```conda create --name env_name```

Activate it with:

```conda activate conda env_name```

And within the same environment, clone the repository

```git clone https://github.com/Ye-Yint-Nyo-Hmine/Song-Recognizer.git```

Navigate to the cloned folder

```cd Song-Recognizer```


### Installing requirements
After creating an environment, install libraries by going into the cloned folder and open the terminal. Then do the following:

```pip install -r requirements.txt```

### Running the Code
Finally, run the file "interface.py":

```python interface.py```


## Usage
Our general interface will look like this: 
![image](https://github.com/user-attachments/assets/3d1ecc77-2fd9-4fc0-9f9e-976a64f8d09b)


With this, 

- [x] [you will be able to add songs into the database](###-adding-songs)
- [x] record a sample of the song you are looking for and matched with the database
- [x] [play all songs uploaded](###-playing-songs)
<br/>
<br/>
<br/>
<br/>

### Adding Songs
To add a song, simply click the Add Song button and a file explorer will pop up. Simply choose a .mp3 file and it will get displayed on your screen where you can play it
<br/>
<br/>
<br/>
<br/>

### Playing Songs
To play a song, just click on the song name. To know if a song is playing it will light up green like shown below
![image](https://github.com/user-attachments/assets/bdfb3547-7f3c-4881-b452-5ca35ba17d06)

To stop playing, click on the song again.


## Libraries Used

