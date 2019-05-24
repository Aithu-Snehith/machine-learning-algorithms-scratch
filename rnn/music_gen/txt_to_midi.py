#!/usr/bin/env python
# coding: utf-8

# In[1]:


from music21 import converter, corpus, instrument, midi, note, chord, pitch, stream
import numpy as np


# In[8]:


temp = np.genfromtxt('gen_processed.txt', delimiter='\n', dtype=str)

for i in range(temp.shape[0]):
    temp[i] = temp[i].replace('<', '').replace('>','')
    
print(type(temp[0]))
notes = []
for i in temp:
    notes.append(i.split())

print(notes)


# In[9]:


s = stream.Stream()
for i in range(len(notes)):
    print(notes[i])
    if notes[i][0] == 'music21.note.Note':
        s.append(note.Note(notes[i][1]))
        print(i)
    elif notes[i][0] == 'music21.chord.Chord' :
        temp = [lel for lel in notes[i][1:]]
        print(temp)
        s.append(chord.Chord(temp))


# In[10]:


s.write('midi', fp='Generated.mid')

