# -*- coding: utf-8 -*-
# !/usr/bin/env python


# -*- coding: utf-8 -*-
# !/usr/bin/env python


import fire

from generate import generate


def generate_multi(*cases):

    for case in cases:
        generate(case)
        print('case \'{}\' Done.'.format(case))

    print('Done.')


if __name__ == '__main__':
    fire.Fire(generate_multi)
