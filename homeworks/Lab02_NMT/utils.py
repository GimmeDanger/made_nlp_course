import torch
import os

from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torchtext.vocab import Vectors

from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import WordPunctTokenizer


def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0) #turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:,0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[1:, 0]), TRG_vocab)
    
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()


#     """ Estimates corpora-level BLEU score of model's translations given inp and reference out """
#     translations, _ = model.translate_lines(inp_lines, **flags)
#     # Note: if you experience out-of-memory error, split input lines into batches and translate separately
#     return corpus_bleu([[ref] for ref in out_lines], translations) * 100
def corpus_blue_accuracy(model, trg_vocab, test_iterator):
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output.argmax(dim=-1)

            original_text.extend([get_text(x, trg_vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, trg_vocab) for x in output[1:].detach().cpu().numpy().T])

        # original_text = flatten(original_text)
        # generated_text = flatten(generated_text)

    return corpus_bleu([[text] for text in original_text], generated_text) * 100


def get_translation_sample(model, trg_vocab, test_iterator, cnt):
    batch = next(iter(test_iterator))
    for idx in range(cnt):
        src = batch.src[:, idx:idx+1]
        trg = batch.trg[:, idx:idx+1]
        generate_translation(src, trg, model, trg_vocab)


def tokenize(x, tokenizer=WordPunctTokenizer()):
    return tokenizer.tokenize(x.lower())


def prepare_data(path_to_data, path_to_src_emb=None, verbose=True):
    # initialize fields
    SRC = Field(tokenize=tokenize,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    TRG = Field(tokenize=tokenize,
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)

    # create dataset and split
    dataset = TabularDataset(
        path=path_to_data,
        format='tsv',
        fields=[('trg', TRG), ('src', SRC)]
    )
    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])

    if path_to_src_emb is not None:
        # load emb vectors: ru emb -- from DeepPavlov
        #                   en emb -- from torch pre
        cache = '.vector_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        enc_vectors = Vectors(name=path_to_src_emb, cache=cache)
        dec_vectors = 'fasttext.simple.300d'

        # init vocabs with emb vectors
        SRC.build_vocab(train_data, min_freq = 3, vectors=enc_vectors)
        TRG.build_vocab(train_data, min_freq = 3, vectors=dec_vectors)
    else:
        SRC.build_vocab(train_data, min_freq = 3)
        TRG.build_vocab(train_data, min_freq = 3)

    if verbose:
        print(f"\nNumber of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(valid_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")
        print(f"Unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
        print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    return train_data, valid_data, test_data, SRC, TRG


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
