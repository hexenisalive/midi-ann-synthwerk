# midi-ann-synthwerk
Python-based (3.6.8) music generation software

###### Coding environment and specifications used durning development:
- OS: Win 8.1 64bit
- IDE: PyCharm Community 2020.1
- Additional compilers: Nvidia CUDA compilation tools 10.1
- GPU: Nvidia GTX 960m

###### Installation:
Program's list of required modules is inside **./requirements.txt**.
Installing newer versions of TensorFlow might result in difficulties while loading previously saved models from **./models/exemplary model**, **./final_models** and **./archive/archive_models**.

###### Usage:
Program promts the user with "yes or no" questions allowing to proceed with the process or to skip it.
Building new dictionaries requires the user to change **dat_loc** variable (found in main.py) with the name of desired MIDI-filled directory found in **./MIDIs**.
Building new models requires the user to make use of Keras API for TensorFlow for them to fully customize their network.
Built and trained models can be saved and then stored in **./models** directory to later generate musical sequences, if specified.
Sequence building requires the system to translate numerical values back to MIDI elements, for that purpose the dictionaries are utilized. 
This may result in conflicts and faulty outputs, if dictionaries are combined with models that were not trained on them. 
To fully reload and properly build sequences from past trained models, dictionaries found in pickle files should be overwritten with the proper MIDI training set.
Plotted sequences are not saved and are only stored temporarly.

###### MIDI files:
- Name: Bernd Krueger
- Source: http://www.piano-midi.de
