#!/usr/bin/env python
# coding: utf-8

# Reference: Kaggle notebook on midi package handling (https://www.kaggle.com/wfaria/midi-music-data-extraction-using-music21)

# In[1]:


from music21 import converter, corpus, instrument, midi, note, chord, pitch
import os
import numpy as np


# In[2]:


midi_path = 'beeth/'

def open_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()         
    return midi.translate.midiFileToStream(mf)

stream_list = []
for i in os.listdir(midi_path):
    stream_list.append(open_midi(midi_path + i))


# In[ ]:


data = []

for i in stream_list:
    for j in i.flat.notes:
        data.append(j)

with open('data.txt', 'w') as f:
    for item in data:
        f.write("%s\n" % item)

