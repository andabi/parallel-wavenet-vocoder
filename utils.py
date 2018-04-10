# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
import os


def remove_all_files(path):
    files = glob.glob('{}/*'.format(path))
    for f in files:
        os.remove(f)
