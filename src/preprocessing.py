"""Shared text preprocessing: tokenization + stopwords.

Used by both the TF-IDF embedder and the BM25 scorer so the two halves of
the hybrid retriever always see the same tokens.
"""

import re

# Small English stopword list (stdlib-only, no NLTK download needed).
STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about into through
    during before after above below to from up down in out on off over under
    again further once here there all any both each few more most other some
    such no nor not only own same so than too very can will just don should
    now is are was were be been being have has had having do does did doing
    would could ought i you he she it we they them his her its our their this
    that these those am as of what which who whom whose where why how
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def stem(word: str) -> str:
    """Light suffix stemmer (stdlib-only).

    Conservative rules for English plurals and verb forms. Query and document
    tokens are stemmed identically, which is what makes matching work.
    For production quality, swap in SnowballStemmer (nltk) here -- the
    interface (word -> word) stays the same.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"  # policies -> policy
    if word.endswith("es") and len(word) > 4 and (
        word[-3] in "sxz" or word[-4:-2] in ("sh", "ch")
    ):
        return word[:-2]  # watches -> watch, boxes -> box
    if (
        word.endswith("s")
        and not word.endswith(("ss", "us"))
        and len(word) > 3
    ):
        return word[:-1]  # passwords -> password, days -> day
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]  # running -> run
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]  # stopped -> stop
        return base
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization with stopword removal + stemming."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
