# Better Shazam and Spotify: Song Recognizer Cog*Work Week 1
## Ever hear a song but can’t remember the name? Well download our Python Script.
### Brought to you by We_Love_Bytes
![image](https://github.com/user-attachments/assets/effc65b3-6066-4bcd-8092-b9cadbfa9213)
<br/>
<br/>


### Installing environment
To start with an environment, we suggest using Conda

[WINDOWS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) 

[MAC](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)


### Creating environment
To create the appropriate environment, type the following in your terminal, replace the following ```env_name``` with the preferred name:

```conda create -n env_name python=3.8 ipython jupyter notebook numpy scipy matplotlib pyaudio numba```

```conda install -n env_name -c conda-forge librosa ffmpeg```

Activate it with:

```conda activate env_name```

Once it is activated, download the [Microphone](https://github.com/CogWorksBWSI/Microphone) package by following the instructions.

Once done, navigate out of Microphone and back into original/parent directory and clone the repository

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
![image](https://github.com/user-attachments/assets/98bcb8e3-9f10-4523-b65f-f26b6ebbffbf)



With this, 

- [x] [you will be able to add songs into the database](#-adding-songs)
- [x] [record a sample of the song you are looking for and matched with the database](#-recording-songs-and-finding-songs)
- [x] [play all songs uploaded](#-playing-songs)
<br/>
<br/>

### Adding Songs
To add a song, simply click the Add Song button and a file explorer will pop up. Simply choose a .mp3 file and it will get displayed on your screen where you can play it
<br/>
<br/>


### Recording Songs and Finding Songs
To record a song, hit the record button which will take input from your microphone for 10 seconds. Then, hit the recognize button to find the best match. (We try out best to get rid of background noise)
<br/>
<br/>


### Playing Songs
To play a song, just click on the song name. To know if a song is playing it will light up green like shown below
![image](https://github.com/user-attachments/assets/7400c985-eff0-41e4-9742-8bd6af00526b)

To stop playing, click on the song again.




## Libraries Used

