# -*- coding: utf-8 -*-
# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import fire

from generate import generate


def generate_multi(*cases):

    for case in cases:
        # TODO fix: define graph seperately.
        generate(case)
        print('case \'{}\' Done.'.format(case))

    print('Done.')


if __name__ == '__main__':
    fire.Fire(generate_multi)
