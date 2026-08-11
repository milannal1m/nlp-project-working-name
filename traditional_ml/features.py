import math

import re

from collections import Counter


import numpy as np

import scipy.sparse as sp

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


from . import config


STOPWORDS = frozenset("""
a an the and or but if while is are was were be been being of to in on at by for
with about against between into through during before after above below from up
down out off over under again further then once here there all any both each few
more most other some such no nor not only own same so than too very s t can will
just don should now this that these those i you he she it we they them his her its
their our your my me him as at has have had do does did having he's it's
""".split())


_WORD_RE = re.compile(r"\b\w+\b")


SPLITTER = "unset"

_PUNKT = None


_JUNK_EXACT = frozenset(["scroll down for video", "scroll down for videos"])

_JUNK_PREFIX = ("click here",)


_SPACED_PERIOD = re.compile(r"\s\.\s+")


def _load_punkt():


    global _PUNKT, SPLITTER

    if _PUNKT is not None or SPLITTER == "regex":

        return

    try:

        import nltk

        try:

            nltk.data.find("tokenizers/punkt_tab")

        except LookupError:

            nltk.download("punkt_tab", quiet=True)

        from nltk.tokenize import sent_tokenize as _nltk_sent

        _nltk_sent("Warm up. The cache.")

        _PUNKT = _nltk_sent

        SPLITTER = "punkt"

    except Exception as e:

        import sys

        print(f"[features] WARNING: punkt unavailable ({e}); falling back to the "

              f"regex sentence splitter. Set SPLITTER=regex expected.", file=sys.stderr, flush=True)

        SPLITTER = "regex"


def _regex_sents(text: str) -> list[str]:

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())

    return [s.strip() for s in parts if s.strip()]


def _is_junk(sentence: str) -> bool:


    low = sentence.strip().lower().rstrip(" .")

    if low in _JUNK_EXACT or low.startswith(_JUNK_PREFIX):

        return True

    return len(_WORD_RE.findall(sentence)) < 4


def _punkt_or_regex(text: str) -> list[str]:

    _load_punkt()

    if _PUNKT is not None:

        return [s.strip() for s in _PUNKT(text.strip()) if s.strip()]

    return _regex_sents(text)


def split_sentences(text: str, dataset: str | None = None) -> list[str]:


    text = text.strip()

    if not text:

        return []

    if dataset == "cnn_dailymail":

        sents: list[str] = []

        for seg in _SPACED_PERIOD.split(text):

            seg = seg.strip()

            if not seg:

                continue

            if seg[-1] not in ".!?\"'":

                seg += "."

            for s in _punkt_or_regex(seg):

                if not _is_junk(s):

                    sents.append(s)

        return sents

    return _punkt_or_regex(text)


def sent_tokenize(text: str) -> list[str]:


    return _regex_sents(text)


def tokenize(text: str) -> list[str]:

    return _WORD_RE.findall(text.lower())


def _ngram_counter(tokens: list[str], n: int) -> Counter:

    if len(tokens) < n:

        return Counter()

    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, cand_total: int, ref_total: int) -> float:

    if cand_total == 0 or ref_total == 0:

        return 0.0

    prec = overlap / cand_total

    rec = overlap / ref_total

    if prec + rec == 0:

        return 0.0

    return 2 * prec * rec / (prec + rec)


def oracle_labels(sentences: list[str], reference: str, max_k: int) -> list[int]:


    n = len(sentences)

    labels = [0] * n

    if n == 0 or not reference.strip():

        return labels


    ref_uni = _ngram_counter(tokenize(reference), 1)

    ref_bi = _ngram_counter(tokenize(reference), 2)

    ref_uni_total = sum(ref_uni.values())

    ref_bi_total = sum(ref_bi.values())

    if ref_uni_total == 0:

        return labels


    sent_uni = [_ngram_counter(tokenize(s), 1) for s in sentences]

    sent_bi = [_ngram_counter(tokenize(s), 2) for s in sentences]


    selected: list[int] = []

    cur_uni: Counter = Counter()

    cur_bi: Counter = Counter()

    cur_score = 0.0


    while len(selected) < min(max_k, n):

        best_gain = 0.0

        best_i = -1

        best_combo = None

        for i in range(n):

            if labels[i]:

                continue

            combo_uni = cur_uni + sent_uni[i]

            combo_bi = cur_bi + sent_bi[i]

            f1_1 = _f1(sum((combo_uni & ref_uni).values()), sum(combo_uni.values()), ref_uni_total)

            f1_2 = _f1(sum((combo_bi & ref_bi).values()), sum(combo_bi.values()), ref_bi_total)

            score = 0.5 * (f1_1 + f1_2)

            if score - cur_score > best_gain + 1e-12:

                best_gain = score - cur_score

                best_i = i

                best_combo = (combo_uni, combo_bi, score)

        if best_i < 0:

            break

        labels[best_i] = 1

        selected.append(best_i)

        cur_uni, cur_bi, cur_score = best_combo


    return labels


DENSE_FEATURE_NAMES = [

    "rel_pos", "abs_pos_norm", "is_first", "in_first_3",

    "word_norm", "len_vs_mean", "tfidf_centroid_cos", "unigram_overlap_rest",

    "numeral_ratio", "cap_ratio", "quote_flag", "stopword_ratio",

    "log_doc_sents", "log_doc_words",

]

DENSE_DIM = len(DENSE_FEATURE_NAMES)


def _centroid_cosine(sentences: list[str]) -> np.ndarray:


    n = len(sentences)

    if n == 0:

        return np.zeros(0, dtype=np.float32)

    if n == 1:

        return np.ones(1, dtype=np.float32)

    try:

        tfidf = TfidfVectorizer().fit_transform(sentences)

    except ValueError:


        return np.zeros(n, dtype=np.float32)

    centroid = np.asarray(tfidf.mean(axis=0)).ravel()

    cnorm = np.linalg.norm(centroid)

    if cnorm == 0:

        return np.zeros(n, dtype=np.float32)


    sims = (tfidf @ (centroid / cnorm))

    return np.asarray(sims).ravel().astype(np.float32)


def dense_features(sentences: list[str]) -> np.ndarray:


    n = len(sentences)

    if n == 0:

        return np.zeros((0, DENSE_DIM), dtype=np.float32)


    toks = [tokenize(s) for s in sentences]

    tok_sets = [set(t) for t in toks]

    wcs = [len(t) for t in toks]

    total_words = sum(wcs) or 1

    mean_wc = total_words / n

    cos = _centroid_cosine(sentences)


    log_sents = min(math.log1p(n) / math.log1p(50), 1.0)

    log_words = min(math.log1p(total_words) / math.log1p(3000), 1.0)


    feats = np.zeros((n, DENSE_DIM), dtype=np.float32)

    for i, s in enumerate(sentences):

        wc = wcs[i] or 1

        rest = set().union(*[tok_sets[j] for j in range(n) if j != i]) if n > 1 else set()

        overlap_rest = (len(tok_sets[i] & rest) / len(tok_sets[i])) if tok_sets[i] else 0.0

        numerals = sum(1 for t in toks[i] if t.isdigit())

        caps = sum(1 for w in s.split() if w[:1].isupper())

        stops = sum(1 for t in toks[i] if t in STOPWORDS)


        feats[i] = (

            i / max(n - 1, 1),

            min(i, 20) / 20.0,

            1.0 if i == 0 else 0.0,

            1.0 if i < 3 else 0.0,

            min(wc, 50) / 50.0,

            min(wc / mean_wc, 3.0) / 3.0,

            float(cos[i]),

            overlap_rest,

            min(numerals / wc, 1.0),

            min(caps / max(len(s.split()), 1), 1.0),

            1.0 if ('"' in s or "'" in s or "“" in s) else 0.0,

            stops / wc,

            log_sents,

            log_words,

        )

    return feats


def build_hashing_vectorizer(normalize: bool = False) -> HashingVectorizer:


    return HashingVectorizer(

        n_features=config.HASH_DIM,

        ngram_range=(1, 2),

        alternate_sign=False,

        norm="l2" if normalize else None,

    )


_VECS: dict = {}


def _get_vectorizer(normalize: bool) -> HashingVectorizer:

    if normalize not in _VECS:

        _VECS[normalize] = build_hashing_vectorizer(normalize=normalize)

    return _VECS[normalize]


def build_model_matrix(name: str, sentences: list[str], dense: np.ndarray | None = None):


    if dense is None:

        dense = dense_features(sentences)

    if not config.USES_SPARSE[name]:

        return dense

    normalize = (name == "logreg")

    sparse = _get_vectorizer(normalize).transform(sentences)

    return sp.hstack([sp.csr_matrix(dense), sparse]).tocsr()


def _trigrams(text: str) -> set:

    toks = tokenize(text)

    if len(toks) < 3:

        return set()

    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)}


def _trigram_jaccard(a: set, b: set) -> float:

    if not a or not b:

        return 0.0

    inter = len(a & b)

    return inter / (len(a) + len(b) - inter)


def select_sentences_v2(scores, sentences: list[str], policy: dict) -> list[int]:


    n = len(sentences)

    if n == 0:

        return []

    mode = policy.get("mode", "topk")

    redundancy = policy.get("redundancy", "none")

    k = policy.get("k")

    budget = policy.get("budget")

    lam = policy.get("lambda", 0.5)


    tris = [_trigrams(s) for s in sentences]

    wcs = [len(_WORD_RE.findall(s)) for s in sentences]

    selected: list[int] = []

    seen_tri: set = set()

    total_words = 0


    def _limit_reached() -> bool:

        return len(selected) >= k if mode == "topk" else total_words >= budget


    if redundancy == "mmr":

        remaining = set(range(n))

        while remaining and not _limit_reached():

            if selected:

                i = max(remaining, key=lambda j: lam * scores[j] - (1 - lam)

                        * max(_trigram_jaccard(tris[j], tris[s]) for s in selected))

            else:

                i = max(remaining, key=lambda j: scores[j])

            remaining.discard(i)

            if mode == "budget" and selected and total_words + wcs[i] > budget:

                continue

            selected.append(i); seen_tri |= tris[i]; total_words += wcs[i]

    else:


        order = sorted(range(n), key=lambda i: scores[i], reverse=True)

        for i in order:

            if _limit_reached():

                break

            if redundancy == "block" and selected and tris[i] and (tris[i] & seen_tri):

                continue

            if mode == "budget" and selected and total_words + wcs[i] > budget:

                continue

            selected.append(i); seen_tri |= tris[i]; total_words += wcs[i]


    return sorted(selected)
