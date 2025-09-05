import os
import re
import numpy as np
import zarr
from numcodecs import Blosc
import h5py
from termcolor import colored
from halo import Halo


compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)


def data_is_valid(data):
    return np.all([item is not None for item in data])


def is_compressible_tensor(data, key):
    if len(data[key]) == 0:
        return False
    try:
        if type(data[key][0][0]) is np.ndarray:
            return True
    except:
        return False


def get_total_bytes_and_bytes_stored(info):

    total = info.obj.info_items()[8][1]
    written = info.obj.info_items()[9][1]

    total_ints = re.findall(r'\d+', total)[0]
    written_ints = re.findall(r'\d+', written)[0]

    return total_ints, written_ints


class DataWriter:
    def __init__(self, directory: str, verbose=True):
        self.directory = directory
        self.counter = 0  # Initialize the frame counter
        # Create the directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.data = {}
        self.current_storage = None
        self.current = None

        self.verbose = verbose

        self.total_bytes = -1
        self.total_stored = -1

    def start(self, i):
        """ This starts the file and actually does the creation """
        name = os.path.join(self.directory, "data%08d.h5" % i)
        self.current = h5py.File(name, 'w')
        self.data = {}
        # self.current_storage = zarr.DirectoryStore(name)
        # self.current = zarr.group(store=self.current_storage)

    def add_info(self, **data):
        """ This writes one-time "header" information """
        if self.current is None:
            raise RuntimeError('no file')
        for k, v in data.items():
            if k not in self.data:
                self.data[k] = v
            else:
                raise RuntimeError('cannot add info for ' + str(k)
                                   + ' as it already exists')

    def finish(self, metadata=None):

        spinner = Halo(text='Writing content to h5 file...', spinner='dots')
        spinner.start()

        """ Convert everything and dump to h5 file """
        for k in self.data.keys():
            if self.data[k] is None:
                raise RuntimeError('data was not properly populated')
            try:
                if is_compressible_tensor(self.data, k):
                    chunk_shape = list(np.array(self.data[k]).shape)
                    chunk_shape[0] = 1

                    compressed = zarr.array(np.array(self.data[k]), chunks=chunk_shape, compressor=compressor)
                    self.track_compression_states(compressed.info)

                    self.current.create_dataset(k, data=compressed)
                elif data_is_valid(self.data[k]):
                    self.current.create_dataset(k, data=self.data[k])
            except TypeError as e:
                print("Failure on key", k)
                print(self.data[k])
                print(e)
                raise e

        if metadata is not None:
            self.add_attributes(**metadata)
        self.current.close()
        self.current = None

        if self.verbose:
            print()
            print(colored("-" * 50, "blue"))
            print(colored("Finished writing file", "blue"))
            print(colored(f"Total data: {round(self.total_bytes / 1_000_000, 3)} MB", "blue"))
            print(colored(f"Total stored: {round(self.total_stored / 1_000_000, 3)} MB", "blue"))
            print(colored(f"Compression ratio: {round(self.total_bytes / self.total_stored, 3)}", "blue"))
            print(colored("-" * 50, "blue"))

        spinner.stop_and_persist(symbol="✅", text="h5 file written to disk!")

    def track_compression_states(self, info):
        try:
            total, written = get_total_bytes_and_bytes_stored(info)
        except:
            total, written = 0, 0
        self.total_bytes += int(total)
        self.total_stored += int(written)

    def add_attributes(self, **data):
        if self.current is None:
            raise RuntimeError('no file')
        # self.current.attrs.update(data)

        for k, v in data.items():
            self.current.attrs[k] = v

    def save_observation(self, data):
        """
        Saves the RGB and depth images to the specified directory with a frame ID matching the counter.

        :param data:
        """

        if self.current is None:
            raise RuntimeError('no file')
        for k, v in data.items():
            if type(v) is dict:
                for sub_k, sub_v in v.items():
                    if sub_k not in self.data:
                        self.data[sub_k] = []
                    self.data[sub_k].append(sub_v)
            else:
                if k not in self.data:
                    self.data[k] = []
                self.data[k].append(v)
