"""Is a NIfTI on disk actually complete?

A killed writer leaves a VALID gzip holding a SHORT image: Python closes the
gzip stream cleanly on the way out, so the container's CRC is fine while most
of the data is missing. Both `path.exists()` and `gzip -t` pass on such a file.

That is how sub-03_ses-02 (240 volumes in the header, 44 on disk) was skipped by
the denoiser as "output exists" and then crashed the extraction downstream.

The gzip trailer's last 4 bytes store the uncompressed size, so we can compare
it against what the NIfTI header implies without decompressing anything.
"""

import struct
from pathlib import Path

import numpy as np
import nibabel as nib


def is_complete(path) -> bool:
    """True if the file holds every volume its header claims."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        img = nib.load(str(path))
        expected = (int(img.dataobj.offset)
                    + int(np.prod(img.shape)) * img.get_data_dtype().itemsize)
        with open(path, "rb") as fh:
            fh.seek(-4, 2)                       # gzip ISIZE, uncompressed size
            actual = struct.unpack("<I", fh.read(4))[0]
        return actual == expected % 2**32        # ISIZE is mod 2**32
    except Exception:
        return False                             # unreadable == not usable
