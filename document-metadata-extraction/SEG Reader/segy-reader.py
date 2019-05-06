import segpy
import numpy
import sys

from segpy.reader import create_reader
from segpy.writer import write_segy

with open(sys.argv[1], 'rb') as segy_in_file:
    """Read and display some properties/metadata of a segy file"""

    # The seg_y_dataset is a lazy-reader, so keep the file open throughout.
    seg_y_dataset = create_reader(segy_in_file)  # Non-standard Rev 1 little-endian

    # print all attributes of the object
    print(seg_y_dataset.__dict__)

    # selected properties to display
    properties = ['num_traces', 'dimensionality', 'data_sample_format', 'data_sample_format_description', '_revision', '_bytes_per_sample', '_max_num_trace_samples']

    # print each selected attribute name and its value if it exists
    for p in properties: 
        if hasattr(seg_y_dataset, p):                   # if the object has the requested attribute
            a = getattr(seg_y_dataset, p)               # get the attributes value
            print(p, a() if callable(a) else a)         # print name and value
        else:  
            print(p, 'is not an attribute')             # if attribute does not exists

    print(seg_y_dataset.trace_header(0))                # print sample header data

    #print('', seg_y_dataset.())