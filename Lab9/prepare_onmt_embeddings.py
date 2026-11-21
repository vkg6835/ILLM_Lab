"""Align projected vectors with OpenNMT's actual source/target token IDs."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from onmt.bin.train import _get_parser
from onmt.inputters.inputter import build_vocab
from onmt.utils.parse import ArgumentParser

config = Path(sys.argv[1]).resolve()
options = _get_parser().parse_args(['-config', str(config)])
ArgumentParser.validate_prepare_opts(options)
vocabs = build_vocab(options, {'src': [], 'tgt': []})
summary = {}
for side, language, path in [('src', 'en', options.pre_word_vecs_enc),
                             ('tgt', 'hi', options.pre_word_vecs_dec)]:
    arrays = np.load(config.parent / 'models' / f'projected_{language}.npz')
    vectors = dict(zip(arrays['tokens'].tolist(), arrays['vectors']))
    tokens = vocabs[side].ids_to_tokens
    assert not any('\r' in token or '\t' in token for token in tokens)
    rng = np.random.default_rng(42)
    matrix = rng.normal(0, .02, size=(len(tokens), 128)).astype('float32')
    matched = 0
    for index, token in enumerate(tokens):
        if token in vectors:
            matrix[index] = vectors[token]
            matched += 1
        if token == '<blank>':
            matrix[index] = 0
    torch.save(torch.from_numpy(matrix), path)
    counts = Counter((config.parent / 'data' / f'train.{language}').read_text(encoding='utf-8').split())
    known = set(tokens)
    token_coverage = sum(n for token, n in counts.items() if token in known) / sum(counts.values())
    pretrained_coverage = sum(n for token, n in counts.items() if token in vectors and token in known) / sum(counts.values())
    summary[side] = {'vocab_size': len(tokens), 'pretrained_types': matched,
                     'training_token_vocabulary_coverage': token_coverage,
                     'training_token_pretrained_coverage': pretrained_coverage,
                     'training_unknown_token_rate': sum(n for token, n in counts.items()
                        if token not in known or token == '<unk>') / sum(counts.values()),
                     'aligned_shape': list(matrix.shape)}
    saved = torch.load(path)
    assert np.array_equal(saved.numpy(), matrix)
    for index, token in enumerate(tokens):
        if token in vectors:
            assert np.array_equal(saved[index].numpy(), vectors[token])
(config.parent / 'embedding_coverage.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary))
