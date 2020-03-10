import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional


class Conll_selection_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

        self.bio_vocab = {}
        self.word_vocab = Counter()
        self.relation_vocab_set = set()
        self.relation_vocab_dict = None

        self._one_pass_train()

        # self.load_prebuilt_word_embedding(hyper.pre_trained_word_emb, 300)

    def _one_pass_train(self):
        # prepare for word_vocab, relation_vocab
        train_path = os.path.join(self.raw_data_root, self.hyper.train)
        self.relation_vocab_set = set()
        sent = []
        dic = {}

        with open(train_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    if sent != []:
                        self.word_vocab.update(sent)
                    sent = []
                    dic = {}
                else:

                    assert len(self._parse_line(line)) == 5

                    num, word, etype, relation, head_list = self._parse_line(
                        line)

                    head_list = eval(head_list)
                    relation = eval(relation)
                    sent.append(word)
                    if relation != ['N']:
                        self.relation_vocab_set.update(relation)
                        for r, h in zip(relation, head_list):
                            dic[word+'~'+r] = h
            self.word_vocab.update(sent)

    def _gen_one_data(self, dataset):
        sent = []
        bio = []
        dic = {}
        selection = []
        selection_dics = []  # temp
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, 'r') as s, open(target, 'w') as t:
            for line in s:
                if line.startswith('#'):
                    if sent != []:
                        triplets = self._process_sent(sent, selection_dics, bio)
                        result = {'text': sent, 'spo_list': triplets,
                                  'bio': bio, 'selection': selection_dics}
                        if len(sent) <= self.hyper.max_text_len:
                            t.write(json.dumps(result))
                            t.write('\n')
                    sent = []
                    bio = []
                    dic = {}  # temp
                    selection_dics = []  # temp
                    selection = []
                else:

                    assert len(self._parse_line(line)) == 5

                    num, word, etype, relation, head_list = self._parse_line(
                        line)

                    head_list = eval(head_list)
                    relation = eval(relation)
                    sent.append(word)
                    bio.append(etype[0])  # only BIO
                    if relation != ['N']:
                        self.relation_vocab_set.update(relation)
                        for r, h in zip(relation, head_list):
                            dic[word+'~'+r] = h
                            selection_dics.append(
                                {'subject': int(num), 'predicate': self.relation_vocab_dict[r], 'object': h})
            if len(sent) <= self.hyper.max_text_len:
                triplets = self._process_sent(sent, selection_dics, bio)
                result = {'text': sent, 'spo_list': triplets,
                          'bio': bio, 'selection': selection_dics}
                t.write(json.dumps(result))

    def load_prebuilt_word_embedding(self, embedding_path, embedding_dim, kept_words=set()):

        lower_kept_words = set([w.lower() for w in kept_words])

        if embedding_path is not None and len(embedding_path) > 0:
            count = 0
            for line in open(embedding_path, "r").readlines():
                count += 1
                if count % 1000 == 0:
                    print("\tRead {0} lines".format(count))

                line = line.strip()
                if not line or len(line) < embedding_dim:
                    continue
                else:
                    word_embedding = line.strip().split(" ")
                    if len(word_embedding) != 1 + embedding_dim:
                        continue
                    word = word_embedding[0]
                    # if word not in self.word_vocab.keys():
                    #    self.word_vocab[word]=len(self.word_vocab)
                    # embedding = [float(val) for val in word_embedding[1:]]
                    self.word_vocab.update(word)


    def gen_all_data(self):
        self._gen_one_data(self.hyper.train)
        self._gen_one_data(self.hyper.dev)

    def gen_bio_vocab(self):
        result = {'<pad>': 3, 'B': 0, 'I': 1, 'O': 2}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w'))

    def gen_relation_vocab(self):
        relation_vocab = {}
        i = 0
        for r in self.relation_vocab_set:
            relation_vocab[r] = i
            i += 1
        relation_vocab['N'] = i
        self.relation_vocab_dict = relation_vocab
        json.dump(relation_vocab,
                  open(self.relation_vocab_path, 'w'),
                  ensure_ascii=True)

    def gen_vocab(self, min_freq: int):
        target = os.path.join(self.data_root, 'word_vocab.json')
        result = {'<pad>': 0}
        i = 1
        for k, v in self.word_vocab.items():
            if v > min_freq:
                result[k] = i
                i += 1
        result['oov'] = i
        json.dump(result, open(target, 'w'), ensure_ascii=False)

    # TODO: fix bug: entity with multiple tokens
    @staticmethod
    def _find_entity(pos, text, sequence_tags):
        entity = []

        if sequence_tags[pos] in ('B', 'O'):
            entity.append(text[pos])
        else:
            temp_entity = []
            while sequence_tags[pos] == 'I':
                temp_entity.append(text[pos])
                pos -= 1
                if pos < 0:
                    break
                if sequence_tags[pos] == 'B':
                    temp_entity.append(text[pos])
                    break
            entity = list(reversed(temp_entity))
        return entity

    def _process_sent(self, sent: List[str], dic: List[Dict[str, int]], bio: List[str]) -> Set[str]:
        id2relation = {v: k for k, v in self.relation_vocab_dict.items()}
        result = []
        for triplets_id in dic:
            s, p, o = triplets_id['subject'], triplets_id['predicate'], triplets_id['object']
            p = id2relation[p]
            s = self._find_entity(s, sent, bio)
            o = self._find_entity(o, sent, bio)

            result.append({'subject': s, 'predicate': p, 'object': o})
        return result

    @staticmethod
    def _parse_line(line):
        result = line.split()
        if len(result) == 5:
            return result
        else:
            a, b, c = result[:3]
            de = result[3:]
            d, e = [], []
            cur = d
            for t in de:
                cur.append(t)
                if t.endswith(']'):
                    cur = e
            return a, b, c, ''.join(d), ''.join(e)


def load_prebuilt_word_embedding(embedding_path, embedding_dim, kept_words=set()):

    word_embedding_dict = dict()
    lower_kept_words = set([w.lower() for w in kept_words])

    if embedding_path is not None and len(embedding_path) > 0:
        count = 0
        for line in open(embedding_path, "r").readlines():
            count += 1
            if count % 1000 == 0:
                print("\tRead {0} lines".format(count))

            line = line.strip()
            if not line or len(line) < embedding_dim:
                continue
            else:
                word_embedding = line.strip().split(" ")
                if len(word_embedding) != 1 + embedding_dim:
                    continue
                word = word_embedding[0]
                embedding = [float(val) for val in word_embedding[1:]]
                if word in word_embedding_dict.keys() or (len(kept_words) > 0 and not (word in kept_words) and not (
                    word.lower() in lower_kept_words)):
                    continue
                else:
                    word_embedding_dict[word] = embedding

    return word_embedding_dict

def load_prebuilt_word_vocab(word_vocab_counter, embedding_path, embedding_dim, kept_words=set()):

    # word_embedding_dict = dict()

    word_vocab = dict()
    for word in word_vocab_counter.keys():
        word_vocab[word] = len(word_vocab)

    lower_kept_words = set([w.lower() for w in kept_words])

    if embedding_path is not None and len(embedding_path) > 0:
        count = 0
        for line in open(embedding_path, "r").readlines():
            count += 1
            if count % 1000 == 0:
                print("\tRead {0} lines".format(count))

            line = line.strip()
            if not line or len(line) < embedding_dim:
                continue
            else:
                word_embedding = line.strip().split(" ")
                if len(word_embedding) != 1 + embedding_dim:
                    continue
                word = word_embedding[0]
                if word not in word_vocab.keys():
                    word_vocab[word] = len(word)

    return word_vocab