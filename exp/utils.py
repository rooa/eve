"""utils.py: utilities for running experiments. Some parts of code are directly
ported from Keras examples and TensorFlow tutorials.
"""

import os
import pickle
import re
import numpy as np
from collections import Counter
from functools import reduce

from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


QFILE = {1: 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt',
         2: 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_train.txt',
         3: 'tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_train.txt',
         4: 'tasks_1-20_v1-2/en-10k/qa4_two-arg-relations_train.txt',
         5: 'tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_train.txt',
         6: 'tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt',
         7: 'tasks_1-20_v1-2/en-10k/qa7_counting_train.txt',
         8: 'tasks_1-20_v1-2/en-10k/qa8_lists-sets_train.txt',
         9: 'tasks_1-20_v1-2/en-10k/qa9_simple-negation_train.txt',
         10: 'tasks_1-20_v1-2/en-10k/qa10_indefinite-knowledge_train.txt',
         11: 'tasks_1-20_v1-2/en-10k/qa11_basic-coreference_train.txt',
         12: 'tasks_1-20_v1-2/en-10k/qa12_conjunction_train.txt',
         13: 'tasks_1-20_v1-2/en-10k/qa13_compound-coreference_train.txt',
         14: 'tasks_1-20_v1-2/en-10k/qa14_time-reasoning_train.txt',
         15: 'tasks_1-20_v1-2/en-10k/qa15_basic-deduction_train.txt',
         16: 'tasks_1-20_v1-2/en-10k/qa16_basic-induction_train.txt',
         17: 'tasks_1-20_v1-2/en-10k/qa17_positional-reasoning_train.txt',
         18: 'tasks_1-20_v1-2/en-10k/qa18_size-reasoning_train.txt',
         19: 'tasks_1-20_v1-2/en-10k/qa19_path-finding_train.txt',
         20: 'tasks_1-20_v1-2/en-10k/qa20_agents-motivations_train.txt'}


def get_subclasses(cls):
    """Get all subclasses (even indirect) of the given class."""
    subclasses = []
    for c in cls.__subclasses__():
        subclasses.append(c)
        subclasses.extend(get_subclasses(c))
    return subclasses


def get_subclass_names(cls):
    """Get the names of all subclasses of the given class."""
    return [c.__name__ for c in get_subclasses(cls)]


def get_subclass_from_name(base_cls, cls_name):
    """Get a subclass given its name."""
    for c in get_subclasses(base_cls):
        if c.__name__ == cls_name:
            return c
    print(base_cls.__subclasses__())
    raise RuntimeError("No such subclass of {}: {}".format(base_cls, cls_name))


def build_subclass_object(base_cls, cls_name, kwargs):
    """Build an object of the named subclass."""
    return get_subclass_from_name(base_cls, cls_name)(**kwargs)


def save_pkl(obj, file_name):
    """Save an object to the given file name as pickle."""
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


def im_bn_axis():
    """Get the axis which should be used for batch normalization with
    image inputs."""
    if K.image_data_format() == "channels_last":
        return 3
    else:
        return 1


def tokenize(sent):
    """Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split(r"(\W+)?", sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbi tasks format. If only_supporting is
    true, only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    """Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story. If max_length
    is supplied, any stories longer than max_length tokens will be discarded.
    """
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    """Pad data with appropriate length."""
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)

    X = pad_sequences(X, maxlen=story_maxlen)
    Xq = pad_sequences(Xq, maxlen=query_maxlen)
    return X, Xq, np.array(Y)


def _read_words(filename):
    with open(filename) as f:
        return f.read().replace('\n', '< e o s >').split()


def _build_vocab(filename):
    data = _read_words(filename)
    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((i, v) for v, i in word_to_id.items())
    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_raw_data(data_path):
    """Load PTB data from data/ptb directory, and numericalize them."""
    train_path = os.path.join(data_path, "ptb.char.train.txt")
    valid_path = os.path.join(data_path, "ptb.char.valid.txt")
    word_to_id, id_to_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    return word_to_id, id_to_word, train_data, valid_data # , test_data


def get_ptb(raw_data, batch_size, num_steps, vocab_size):
    """Returns batch generator for x and y."""
    data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    nstep_len = data_len // num_steps
    x = data[:nstep_len*num_steps].reshape(-1, num_steps)
    y = data[1:nstep_len*num_steps+1].reshape(-1, num_steps)

    # Shuffle the data.
    combined = np.hstack((x, y))
    np.random.shuffle(combined)
    x, y = np.hsplit(combined, [num_steps])
    x = np.array([to_categorical(xx, num_classes=vocab_size) for xx in x])
    y = np.array([to_categorical(yy, num_classes=vocab_size) for yy in y])

    return x, y
