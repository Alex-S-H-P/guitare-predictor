import jams
import os
from codebase import path_handler
import matplotlib.pyplot as plt
import random
import codebase.interpreter as itp


if __name__ == '__main__':
    random.seed(1)
    path = path_handler.path_to_specific_dataset("annotations")
    choices = os.listdir(path)
    idx = random.randrange(len(choices))
    print("\033[33;1m",
          "CHOSE file n°", "\033[36;1m",
          f"{idx}", "\033[33;1m : \033[32;1m",
          f"{choices[idx]}"
          "\033[0m", sep="")
    path = path + ("/" if not path.endswith("/") else "") + choices[idx]
    del choices
    del idx
    print("\033[33;1m",
          "OPENING",
          "\033[36;1m",
          f"{path}",
          "\033[0m")
    jam = jams.load(path)
    instructed_chords = jam.search(namespace='chord')[0]
    performed_chords = jam.search(namespace='chord')[1]
    fig = plt.figure(figsize=(8, 7), dpi=300)
    ax1 = plt.subplot(211)
    itp.visualize_jams_pt(jam)  # pt for pitch track
    ax2 = plt.subplot(212, sharex=ax1)
    itp.tablaturize_jams(jam)
    plt.xlim(4.9, 10)  # this is the time window in seconds that I'm plotting

    itp.add_annotations(ax1, performed_chords, ygrow_ratio=0.04, label_xoffset=0.06, label_yoffset=2)
    itp.add_annotations(ax2, instructed_chords, ygrow_ratio=0.2, label_xoffset=0.06, label_yoffset=-0.4)
    plt.show()
    print("\033[33;1m",
          "DONE",
          "\033[0m")
