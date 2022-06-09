"""

Author : Adam ALIOUAT https://github.com/adam-at/
"""

import jams
import os
import matplotlib.pyplot as plt

import interpreter as itp

gs_path = "F:/PycharmProjects/guitare-predictor/dataset/"
anno_dir = "annotations/"
excerpt = "05_Jazz2-187-F#_comp.jams"

plt.rcParams.update({'font.size': 5})

jam = jams.load(os.path.join(gs_path+anno_dir, excerpt))
instructed_chords = jam.search(namespace='chord')[0]
# performed_chords = jam.search(namespace='chord')[1]
fig = plt.figure(figsize=(5,5), dpi=200)
ax1 = plt.subplot(211)
itp.visualize_jams_pt(jam) #pt for pitch track
ax2 = plt.subplot(212, sharex=ax1)
itp.tablaturize_jams(jam)
plt.xlim(4.9, 10) # time window

itp.add_annotations(ax1, instructed_chords, ygrow_ratio=0.04, label_xoffset=0.06, label_yoffset=2)
itp.add_annotations(ax2, instructed_chords, ygrow_ratio=0.2, label_xoffset=0.06, label_yoffset=-0.4)
plt.show()
