#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""quantulum tests."""

# Standard library
import json
import os
import re
import unittest

# Dependencies
import wikipedia

from quantulum import classes
# Quantulum
from quantulum import load
from quantulum import parser

COLOR1 = '\033[94m%s\033[0m'
COLOR2 = '\033[91m%s\033[0m'
TOPDIR = os.path.dirname(__file__) or "."


def embed_text(quants, beg_char, chunk, content):
    """Embed quantities in text."""
    if quants:
        end_char = max((chunk + 1) * 1000, quants[-1].span[1])
        text = content[beg_char:end_char]
        shift = 0
        for quantity in quants:
            index = quantity.span[1] - beg_char + shift
            to_add = COLOR1 % (' {' + str(quantity) + '}')
            text = text[0:index] + to_add + COLOR2 % text[index:]
            shift += len(to_add) + len(COLOR2) - 6
    else:
        end_char = (chunk + 1) * 1000
        text = content[beg_char:end_char]

    return text, end_char


def wiki_test(page='CERN'):
    """Download a wikipedia page and test the parser on its content.

    Pages full of units:
        CERN
        Hubble_Space_Telescope,
        Herschel_Space_Observatory
    """
    content = wikipedia.page(page).content
    parsed = parser.parse(content)
    parts = int(round(len(content) * 1.0 / 1000))

    end_char = 0
    for num, chunk in enumerate(range(parts)):
        _ = os.system('clear')
        quants = [j for j in parsed if chunk * 1000 < j.span[0] < (chunk + 1) *
                  1000]
        beg_char = max(chunk * 1000, end_char)
        text, end_char = embed_text(quants, beg_char, chunk, content)
        print(COLOR2 % text)
        try:
            _ = input('--------- End part %d of %d\n' % (num + 1, parts))
        except (KeyboardInterrupt, EOFError):
            return


def get_quantity(test, item):
    """Build a single quantity for the test."""
    try:
        unit = load.NAMES[item['unit']]
    except KeyError:
        try:
            entity = item['entity']
        except KeyError:
            print('Could not find %s, provide "dimensions" and'
                  ' "entity"' % item['unit'])
            return
        if entity == 'unknown':
            dimensions = [{'base': load.NAMES[i['base']].entity.name,
                           'power': i['power']} for i in
                          item['dimensions']]
            entity = classes.Entity(name='unknown', dimensions=dimensions)
        elif entity in load.ENTITIES:
            entity = load.ENTITIES[entity]
        else:
            print('Could not find %s, provide "dimensions" and'
                  ' "entity"' % item['unit'])
            return
        unit = classes.Unit(name=item['unit'],
                            dimensions=item['dimensions'],
                            entity=entity)
    try:
        span = next(
            re.finditer(re.escape(item['surface']), test['req'])).span()
    except StopIteration:
        print('Surface mismatch for "%s"' % test['req'])
        return

    uncert = None
    if 'uncertainty' in item:
        uncert = item['uncertainty']

    quantity = classes.Quantity(value=item['value'],
                                unit=unit,
                                surface=item['surface'],
                                span=span,
                                uncertainty=uncert)

    return quantity


def load_tests_from_json():
    """Load all tests from tests.json."""
    path = os.path.join(TOPDIR, 'tests.json')
    with open(path, encoding='utf-8') as f:
        tests = json.load(f)

    for test in tests:
        res = []
        for item in test['res']:
            quantity = get_quantity(test, item)
            if quantity is None:
                return
            res.append(quantity)
        test['res'] = [i for i in res]

    return tests


class EndToEndTests(unittest.TestCase):
    """Test suite for the quantulum project."""

    def test_load_tests(self):
        """Test for tests.load_test() function."""
        self.assertFalse(load_tests_from_json() is None)

    def test_parse(self):
        result = {'passed': 0, 'not': 0}

        """Test for parser.parse() function."""
        all_tests = load_tests_from_json()
        for test in sorted(all_tests, key=lambda x: len(x['req'])):
            try:
                self.assertEqual(parser.parse(test['req']),
                                 test['res'])
            except AssertionError:
                result['not'] = result['not'] + 1
            result['passed'] = result['passed'] + 1
        print("Passed: {}, not passed: {}".format(result['passed'],
                                                  result['not']))


if __name__ == '__main__':
    unittest.main()
