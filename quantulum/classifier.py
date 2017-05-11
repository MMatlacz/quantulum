# -*- coding: utf-8 -*-

from __future__ import print_function

from typing import Dict, Tuple

"""quantulum classifier functions."""

# Standard library
import json
import logging
import os
import pickle
import re

# Dependencies
import wikipedia
from stemming.porter2 import stem

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    USE_CLF = True
except ImportError:
    USE_CLF = False

# Quantulum
from quantulum import load


def download_wiki() -> None:
    """Download WikiPedia pages of ambiguous units."""
    ambiguous = [i for i in load.UNITS.items() if len(i[1]) > 1]
    ambiguous += [i for i in load.DERIVED_ENT.items() if len(i[1]) > 1]
    pages = set([(j.name, j.uri) for i in ambiguous for j in i[1]])

    objs = []
    for num, page in enumerate(pages):
        obj = {'url': page[1]}
        obj['_id'] = obj['url'].replace('https://en.wikipedia.org/wiki/', '')
        obj['clean'] = obj['_id'].replace('_', ' ')

        print('---> Downloading %s (%d of %d)' % \
              (obj['clean'], num + 1, len(pages)))

        obj['text'] = wikipedia.page(obj['clean']).content
        obj['unit'] = page[0]
        objs.append(obj)

    path = os.path.join(load.TOPDIR, 'wiki.json')
    os.remove(path)
    json.dump(objs, open(path, 'w'), indent=4, sort_keys=True)

    print('\n---> All done.\n')


def clean_text(text: str) -> str:
    """Clean text for TFIDF."""
    new_text = re.sub(r'\p{P}+', ' ', text)

    new_text = [stem(i) for i in new_text.lower().split() if not
    re.findall(r'[0-9]', i)]

    new_text = ' '.join(new_text)

    return new_text


def train_classifier(download: bool = True, parameters: Dict = None,
                     ngram_range: Tuple = (1, 1)) -> None:
    """Train the intent classifier."""
    if download:
        download_wiki()

    path = os.path.join(load.TOPDIR, 'train.json')
    training_set = json.load(open(path))
    path = os.path.join(load.TOPDIR, 'wiki.json')
    wiki_set = json.load(open(path))

    target_names = list(set([i['unit'] for i in training_set + wiki_set]))
    train_data, train_target = [], []
    for example in training_set + wiki_set:
        train_data.append(clean_text(example['text']))
        train_target.append(target_names.index(example['unit']))

    tfidf_model = TfidfVectorizer(sublinear_tf=True,
                                  ngram_range=ngram_range,
                                  stop_words='english')

    matrix = tfidf_model.fit_transform(train_data)

    if parameters is None:
        parameters = {'loss': 'log', 'penalty': 'l2', 'n_iter': 50,
                      'alpha': 0.00001, 'fit_intercept': True}

    clf = SGDClassifier(**parameters).fit(matrix, train_target)
    obj = {'tfidf_model': tfidf_model,
           'clf': clf,
           'target_names': target_names}
    path = os.path.join(load.TOPDIR, 'clf.pickle')
    pickle.dump(obj, open(path, 'w'))


def load_classifier() -> Tuple:
    """Train the intent classifier."""
    path = os.path.join(load.TOPDIR, 'clf.pickle')
    obj = pickle.load(open(path))

    return obj['tfidf_model'], obj['clf'], obj['target_names']


if USE_CLF:
    TFIDF_MODEL, CLF, TARGET_NAMES = load_classifier()
else:
    TFIDF_MODEL, CLF, TARGET_NAMES = None, None, None


def disambiguate_entity(key: str, text: str) -> str:
    """Resolve ambiguity between entities with same dimensionality."""
    new_ent = load.DERIVED_ENT[key][0]

    if len(load.DERIVED_ENT[key]) > 1:
        transformed = TFIDF_MODEL.transform([text])
        scores = CLF.predict_proba(transformed).tolist()[0]
        scores = sorted(zip(scores, TARGET_NAMES), key=lambda x: x[0],
                        reverse=True)
        names = [i.name for i in load.DERIVED_ENT[key]]
        scores = [i for i in scores if i[1] in names]
        try:
            new_ent = load.ENTITIES[scores[0][1]]
        except IndexError:
            logging.debug('\tAmbiguity not resolved for "%s"', str(key))

    return new_ent


def disambiguate_unit(unit: str, text: str) -> str:
    """
    Resolve ambiguity.

    Distinguish between units that have same names, symbols or abbreviations.
    """
    new_unit = load.UNITS[unit]
    if not new_unit:
        new_unit = load.LOWER_UNITS[unit.lower()]
        if not new_unit:
            raise KeyError('Could not find unit "%s"' % unit)

    if len(new_unit) > 1:
        transformed = TFIDF_MODEL.transform([clean_text(text)])
        scores = CLF.predict_proba(transformed).tolist()[0]
        scores = sorted(zip(scores, TARGET_NAMES), key=lambda x: x[0],
                        reverse=True)
        names = [i.name for i in new_unit]
        scores = [i for i in scores if i[1] in names]
        try:
            final = load.UNITS[scores[0][1]][0]
            logging.debug('\tAmbiguity resolved for "%s" (%s)', unit, scores)
        except IndexError:
            logging.debug('\tAmbiguity not resolved for "%s"', unit)
            final = new_unit[0]
    else:
        final = new_unit[0]

    return final
