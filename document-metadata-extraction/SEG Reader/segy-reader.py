import segpy
import numpy
import sys

from segpy.reader import create_reader
from segpy.writer import write_segy

print("a")

with open(sys.argv[1], 'rb') as segy_in_file:
    # The seg_y_dataset is a lazy-reader, so keep the file open throughout.
    seg_y_dataset = create_reader(segy_in_file)  # Non-standard Rev 1 little-endian

    print(seg_y_dataset.__dict__)

    properties = ['num_traces', 'dimensionality', 'data_sample_format', 'data_sample_format_description', '_revision', '_bytes_per_sample', '_max_num_trace_samples']

    for p in properties: 
        if hasattr(seg_y_dataset, p):
            a = getattr(seg_y_dataset, p)
            print(p, a() if callable(a) else a)
        else:
            print(p, 'is not an attribute')

    print(seg_y_dataset.trace_header(0))

    #print('', seg_y_dataset.())