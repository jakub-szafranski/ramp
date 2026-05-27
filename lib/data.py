# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for NumPy and PyTorch for reproducibility.
    
    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _sample_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer):
    """Tokenize a text corpus and sample contiguous LM calibration windows."""
    trainenc = tokenizer(" ".join(texts), return_tensors='pt')
    if trainenc.input_ids.shape[1] <= seqlen:
        raise ValueError(
            f"Tokenized corpus too short for seqlen={seqlen} "
            f"(tokens={trainenc.input_ids.shape[1]})."
        )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    testenc = TokenizerWrapper(trainenc.input_ids[:, :(256 * seqlen)])
    return trainloader, testenc

# Load and process PTB (Penn Treebank) dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    """
    Load and process PTB (Penn Treebank) dataset.
    
    Args:
        nsamples (int): Number of samples to extract.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for each sample.
        tokenizer (Tokenizer): Tokenizer to use for text encoding.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test set.
    """
    # Load train and test datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # traindata = load_dataset('text', data_files='datasets/wikitext/wiki.train.raw', split="train")
    # testdata = load_dataset('text', data_files='datasets/wikitext/wiki.test.raw', split="train")
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process C4 (Common Crawl) dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    # traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')
    
    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Load and process GSM8K (grade-school math) dataset
def get_gsm8k(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the GSM8K dataset (grade-school math reasoning).

    Each example is formatted as:
        Question: <question>\\nAnswer: <answer>

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and a dummy test wrapper.
    """
    traindata = load_dataset('openai/gsm8k', 'main', split='train')

    # Format each example as a self-contained QA block
    texts = [
        f"Question: {ex['question']}\nAnswer: {ex['answer']}"
        for ex in traindata
    ]

    trainenc = tokenizer(" ".join(texts), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Use the same training corpus as a dummy test set (no official LM test split)
    testenc = TokenizerWrapper(trainenc.input_ids[:, :(256 * seqlen)])
    return trainloader, testenc


# Load and process SQuAD v2 dataset
def get_squad_v2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the SQuAD v2 dataset (reading comprehension).

    Each example is formatted as:
        <context>
        Question: <question>
        Answer: <answer>   (or "unanswerable" for v2 examples with no answer)

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and a dummy test wrapper.
    """
    traindata = load_dataset('rajpurkar/squad_v2', split='train')

    texts = []
    for ex in traindata:
        answer = ex['answers']['text'][0] if ex['answers']['text'] else 'unanswerable'
        texts.append(
            f"{ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
        )

    trainenc = tokenizer(" ".join(texts), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    testenc = TokenizerWrapper(trainenc.input_ids[:, :(256 * seqlen)])
    return trainloader, testenc


def get_boolq(nsamples, seed, seqlen, tokenizer):
    """Load BoolQ and format as QA text calibration corpus."""
    traindata = load_dataset('google/boolq', split='train')
    texts = [
        f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer: {'yes' if ex['answer'] else 'no'}"
        for ex in traindata
    ]
    return _sample_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer)


def get_winogrande(nsamples, seed, seqlen, tokenizer):
    """Load WinoGrande XL and format as sentence completion text."""
    traindata = load_dataset('allenai/winogrande', 'winogrande_xl', split='train')
    texts = []
    for ex in traindata:
        answer = ex.get('answer')
        if answer not in ('1', '2'):
            continue
        replacement = ex['option1'] if answer == '1' else ex['option2']
        text = ex['sentence'].replace('_', replacement)
        texts.append(f"Sentence: {text}")
    return _sample_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer)


def get_hellaswag(nsamples, seed, seqlen, tokenizer):
    """Load HellaSwag and format as context + gold ending text."""
    traindata = load_dataset('Rowan/hellaswag', split='train')
    texts = []
    for ex in traindata:
        endings = ex.get('endings', [])
        label = ex.get('label')
        if isinstance(label, str):
            if label not in ('0', '1', '2', '3'):
                continue
            label = int(label)
        if not isinstance(label, int) or label < 0 or label >= len(endings):
            continue
        context = f"{ex.get('ctx_a', '')} {ex.get('ctx_b', '')}".strip()
        texts.append(f"Context: {context}\nContinuation: {endings[label]}")
    return _sample_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer)


def get_arc_easy(nsamples, seed, seqlen, tokenizer):
    """Load ARC-Easy and format as question + correct option text."""
    traindata = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train')
    label_map = {letter: i for i, letter in enumerate('ABCDE')}
    label_map.update({str(i + 1): i for i in range(5)})

    texts = []
    for ex in traindata:
        answer_texts = ex['choices']['text']
        answer_key = ex.get('answerKey', '')
        if answer_key not in label_map:
            continue
        gold = label_map[answer_key]
        if gold >= len(answer_texts):
            continue
        texts.append(
            f"Question: {ex['question']}\nAnswer: {answer_texts[gold]}"
        )
    return _sample_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer)


def get_arc_challenge(nsamples, seed, seqlen, tokenizer):
    """Load ARC-Challenge and format as question + correct option text."""
    traindata = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')
    label_map = {letter: i for i, letter in enumerate('ABCDE')}
    label_map.update({str(i + 1): i for i in range(5)})

    texts = []
    for ex in traindata:
        answer_texts = ex['choices']['text']
        answer_key = ex.get('answerKey', '')
        if answer_key not in label_map:
            continue
        gold = label_map[answer_key]
        if gold >= len(answer_texts):
            continue
        texts.append(
            f"Question: {ex['question']}\nAnswer: {answer_texts[gold]}"
        )
    return _sample_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer)


def get_openbookqa(nsamples, seed, seqlen, tokenizer):
    """Load OpenBookQA and format as question + correct option text."""
    traindata = load_dataset('allenai/openbookqa', 'main', split='train')
    label_map = {letter: i for i, letter in enumerate('ABCDE')}
    label_map.update({str(i + 1): i for i in range(5)})

    texts = []
    for ex in traindata:
        answer_texts = ex['choices']['text']
        answer_key = ex.get('answerKey', '')
        if answer_key not in label_map:
            continue
        gold = label_map[answer_key]
        if gold >= len(answer_texts):
            continue
        question = ex.get('question_stem') or ex.get('question')
        if not question:
            continue
        texts.append(f"Question: {question}\nAnswer: {answer_texts[gold]}")
    return _sample_trainloader_from_texts(texts, nsamples, seed, seqlen, tokenizer)


# Function to select the appropriate loader based on dataset name
def get_loaders(name='wikitext2', nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): The name of the dataset ('wikitext2', 'c4', or 'ptb').
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation/test set.
    """
    # Determine which dataset to use based on 'name' parameter and return corresponding loader
    name = name.lower()
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    elif "gsm8k" in name:
        return get_gsm8k(nsamples, seed, seqlen, tokenizer)
    elif "squad" in name:
        return get_squad_v2(nsamples, seed, seqlen, tokenizer)
    elif "boolq" in name:
        return get_boolq(nsamples, seed, seqlen, tokenizer)
    elif "winogrande" in name:
        return get_winogrande(nsamples, seed, seqlen, tokenizer)
    elif "hellaswag" in name:
        return get_hellaswag(nsamples, seed, seqlen, tokenizer)
    elif name in ("arc_easy", "arc-easy"):
        return get_arc_easy(nsamples, seed, seqlen, tokenizer)
    elif name in ("arc_challenge", "arc-challenge", "arc_train"):
        return get_arc_challenge(nsamples, seed, seqlen, tokenizer)
    elif name in ("openbookqa", "openbook_qa", "openbook-qa"):
        return get_openbookqa(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unsupported calibration dataset: {name}")

if __name__ == "__main__": 
    get_loaders('wikitext2', seed=0, seqlen=2048, tokenizer=None)

# Note:
# This script is designed to load and process various text datasets for training language models.
# It includes functions to load PTB (Penn Treebank), Wikitext-2, and C4 (Common Crawl) datasets.
# Each loading function returns a trainloader (list of input and target pairs) and encoded validation/test set.
