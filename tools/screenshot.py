#!/usr/bin/env python3
"""Screenshot a window by title substring (X11, via xwd) -> PNG.  python3 tools/screenshot.py Webots out.png"""
import struct
import subprocess
import sys

import numpy as np
from PIL import Image

title, out = sys.argv[1], sys.argv[2]
tree = subprocess.run(['xwininfo', '-root', '-tree'], capture_output=True, text=True).stdout
wid = next(l.split()[0] for l in tree.splitlines() if title in l and '"' in l and 'mutter' not in l)
d = subprocess.run(['xwd', '-id', wid, '-silent'], capture_output=True).stdout
h = struct.unpack('>25I', d[:100])
hs, w, hh, bpl, ncol = h[0], h[4], h[5], h[12], h[19]
a = np.frombuffer(d[hs + ncol * 12:hs + ncol * 12 + bpl * hh], np.uint8).reshape(hh, bpl)[:, :w * 4].reshape(hh, w, 4)
Image.fromarray(a[:, :, [2, 1, 0]]).resize((w // 2, hh // 2)).save(out)
print(out, w, hh)
