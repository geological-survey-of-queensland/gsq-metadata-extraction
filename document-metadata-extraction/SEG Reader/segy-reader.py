import segpy
import numpy

from segpy.reader import create_reader
from segpy.writer import write_segy

print("a")

with open("2.sgy", 'rb') as segy_in_file:
    # The seg_y_dataset is a lazy-reader, so keep the file open throughout.
    seg_y_dataset = create_reader(segy_in_file)  # Non-standard Rev 1 little-endian
    print(seg_y_dataset.num_traces())