import nltk
import os
import re
import itertools
import collections
import pkg_resources
from io import open

class Rouge:
    DEFAULT_METRICS = ["rouge-n"]
    DEFAULT_N = 1
    STATS = ["f", "p", "r"]
    AVAILABLE_METRICS = {"rouge-n", "rouge-l", "rouge-w", "rouge-s", "rouge-su"}
    AVAILABLE_LENGTH_LIMIT_TYPES = {'words', 'bytes'}
    REMOVE_CHAR_PATTERN = re.compile('[^A-Za-z0-9]')

    # Hack to not tokenize "cannot" to "can not" and consider them different as in the official ROUGE script
    KEEP_CANNOT_IN_ONE_WORD = re.compile('cannot')
    KEEP_CANNOT_IN_ONE_WORD_REVERSED = re.compile('_cannot_')

    WORDNET_KEY_VALUE = {}
    WORDNET_DB_FILEPATH = 'wordnet_key_value.txt'
    WORDNET_DB_FILEPATH_SPECIAL_CASE = 'wordnet_key_value_special_cases.txt'
    WORDNET_DB_DELIMITER = '|'
    STEMMER = None
    STOPWORDS_SET = set()
    STOPWORDS_FILEPATH = 'smart_common_words.txt'

    def __init__(self, metrics=None, max_n=None, max_skip_bigram=4, limit_length=True, length_limit=665, length_limit_type='bytes', apply_avg=True, apply_best=False, stemming=True, stopword_removal=True, alpha=0.5, weight_factor=1.0, ensure_compatibility=True):
        """
        Handle the ROUGE score computation as in the official perl script.

        Note 1: Small differences might happen if the resampling of the perl script is not high enough (as the average depends on this).
        Note 2: Stemming of the official Porter Stemmer of the ROUGE perl script is slightly different and the Porter one implemented in NLTK. However, special cases of DUC 2004 have been traited.
                The solution would be to rewrite the whole perl stemming in python from the original script

        Args:
          metrics: What ROUGE score to compute. Available: ROUGE-N, ROUGE-L, ROUGE-W. Default: ROUGE-N
          max_n: N-grams for ROUGE-N if specify. Default:1
          max_skip_bigram: Maximum allowed gap between swords in a skip bigram for ROUGE-S and ROUGE-SU if specify. Default: 4
          limit_length: If the summaries must be truncated. Defaut:True
          length_limit: Number of the truncation where the unit is express int length_limit_Type. Default:665 (bytes)
          length_limit_type: Unit of length_limit. Available: words, bytes. Default: 'bytes'
          apply_avg: If we should average the score of multiple samples. Default: True. If apply_Avg & apply_best = False, then each ROUGE scores are independant
          apply_best: Take the best instead of the average. Default: False, then each ROUGE scores are independant
          stemming: Apply stemming to summaries. Default: True
          stopword_removal: Remove stopwords from summaries. Default: True
          alpha: Alpha use to compute f1 score: P*R/((1-a)*P + a*R). Default:0.5
          weight_factor: Weight factor to be used for ROUGE-W. Official rouge score defines it at 1.2. Default: 1.0
          ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough). Default:True

        Raises:
          ValueError: raises exception if metric is not among AVAILABLE_METRICS
          ValueError: raises exception if length_limit_type is not among AVAILABLE_LENGTH_LIMIT_TYPES
          ValueError: raises exception if weight_factor < 0
        """
        self.metrics = metrics[:] if metrics is not None else Rouge.DEFAULT_METRICS
        for m in self.metrics:
            if m not in Rouge.AVAILABLE_METRICS:
                raise ValueError("Unknown metric '{}'".format(m))

        self.max_n = max_n if "rouge-n" in self.metrics else None
        # Add all rouge-n metrics
        if self.max_n is not None:
            index_rouge_n = self.metrics.index('rouge-n')
            del self.metrics[index_rouge_n]
            self.metrics += ['rouge-{}'.format(n) for n in range(1, self.max_n + 1)]

        self.max_skip_bigram = max_skip_bigram
        if 'rouge-s' in self.metrics:
            index_rouge_s = self.metrics.index('rouge-s')
            del self.metrics[index_rouge_s]
            self.metrics.append('rouge-s{}'.format(self.max_skip_bigram))
        elif 'rouge-su' in self.metrics:
            index_rouge_su = self.metrics.index('rouge-su')
            del self.metrics[index_rouge_su]
            self.metrics.append('rouge-su{}'.format(self.max_skip_bigram))
        else:
            self.max_skip_bigram = None

        self.metrics = set(self.metrics)

        self.limit_length = limit_length
        if self.limit_length:
            if length_limit_type not in Rouge.AVAILABLE_LENGTH_LIMIT_TYPES:
                raise ValueError("Unknown length_limit_type '{}'".format(length_limit_type))

        self.length_limit = length_limit
        if self.length_limit == 0:
            self.limit_length = False
        self.length_limit_type = length_limit_type
        self.stemming = stemming
        self.stopword_removal = stopword_removal

        self.apply_avg = apply_avg
        self.apply_best = apply_best
        self.alpha = alpha
        self.weight_factor = weight_factor
        if self.weight_factor <= 0:
            raise ValueError("ROUGE-W weight factor must greater than 0.")
        self.ensure_compatibility = ensure_compatibility

        # Load static objects
        if len(Rouge.WORDNET_KEY_VALUE) == 0:
            Rouge.load_wordnet_db(ensure_compatibility)
        if Rouge.STEMMER is None:
            Rouge.load_stemmer(ensure_compatibility)
        if len(Rouge.STOPWORDS_SET) == 0:
            Rouge.load_stopwords()

    @staticmethod
    def load_stemmer(ensure_compatibility):
        """
        Load the stemmer that is going to be used if stemming is enabled
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)
        """
        Rouge.STEMMER = nltk.stem.porter.PorterStemmer('ORIGINAL_ALGORITHM') if ensure_compatibility else nltk.stem.porter.PorterStemmer()

    @staticmethod
    def load_wordnet_db(ensure_compatibility):
        """
        Load WordNet database to apply specific rules instead of stemming + load file for special cases to ensure kind of compatibility (at list with DUC 2004) with the original stemmer used in the Perl script
        Args
            ensure_compatibility: Use same stemmer and special "hacks" to product same results as in the official perl script (besides the number of sampling if not high enough)

        Raises:
            FileNotFoundError: If one of both databases is not found
        """
        files_to_load = [Rouge.WORDNET_DB_FILEPATH]
        if ensure_compatibility:
            files_to_load.append(Rouge.WORDNET_DB_FILEPATH_SPECIAL_CASE)

        for wordnet_db in files_to_load:
            filepath = pkg_resources.resource_filename(__name__, wordnet_db)
            if not os.path.exists(filepath):
                raise FileNotFoundError("The file '{}' does not exist".format(filepath))

            with open(filepath, 'r', encoding='utf-8') as fp:
                for line in fp:
                    k, v = line.strip().split(Rouge.WORDNET_DB_DELIMITER)
                    assert k not in Rouge.WORDNET_KEY_VALUE
                    Rouge.WORDNET_KEY_VALUE[k] = v

    @staticmethod
    def load_stopwords():
        """
        Load stopwords
        Raises:
            FileNotFoundError: If stopwords file is not found
        """
        filepath = pkg_resources.resource_filename(__name__, Rouge.STOPWORDS_FILEPATH)
        if not os.path.exists(filepath):
            raise FileNotFoundError("The file '{}' does not exist".format(filepath))

        with open(filepath, 'r', encoding='utf-8') as fp:
            for line in fp:
                k = line.strip()
                Rouge.STOPWORDS_SET.add(k)

    @staticmethod
    def tokenize_text(text, language='english'):
        """
        Tokenize text in the specific language

        Args:
          text: The string text to tokenize
          language: Language of the text

        Returns:
          List of tokens of text
        """
        return nltk.word_tokenize(text, language)

    @staticmethod
    def split_into_sentences(text, ensure_compatibility, language='english'):
        """
        Split text into sentences, using specified language. Use PunktSentenceTokenizer

        Args:
          text: The string text to tokenize
          ensure_compatibility: Split sentences by '\n' instead of NLTK sentence tokenizer model
          language: Language of the text

        Returns:
          List of tokens of text
        """
        if ensure_compatibility:
            return text.split('\n')
        else:
            return nltk.sent_tokenize(text, language)

    @staticmethod
    def stem_tokens(tokens):
        """
        Apply WordNetDB rules or Stem each token of tokens

        Args:
          tokens: List of tokens to apply WordNetDB rules or to stem

        Returns:
          List of final stems
        """
        # Stemming & Wordnet apply only if token has at least 3 chars
        for i, token in enumerate(tokens):
            if len(token) > 0:
                if len(token) > 3:
                    if token in Rouge.WORDNET_KEY_VALUE:
                        token = Rouge.WORDNET_KEY_VALUE[token]
                    else:
                        token = Rouge.STEMMER.stem(token)
                    tokens[i] = token

        return tokens

    @staticmethod
    def remove_stopwords(tokens):
        """
        Remove stopwords

        Args:
          tokens: List of tokens to apply stopword removal

        Returns:
          List of tokens without stopwords
        """
        for i, token in reversed(list(enumerate(tokens))):
            if token in Rouge.STOPWORDS_SET:
                del tokens[i]

        return tokens

    @staticmethod
    def _get_ngrams(n, text):
        """
        Calcualtes n-grams.

        Args:
          n: which n-grams to calculate
          text: An array of tokens

        Returns:
          A set of n-grams with their number of occurences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        ngram_set = collections.defaultdict(int)
        max_index_ngram_start = len(text) - n
        for i in range(max_index_ngram_start + 1):
            ngram_set[tuple(text[i:i + n])] += 1
        return ngram_set

    @staticmethod
    def _split_into_words(sentences):
        """
        Splits multiple sentences into words and flattens the result

        Args:
          sentences: list of string

        Returns:
          A list of words (split by white space)
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        return list(itertools.chain(*[_.split() for _ in sentences]))

    @staticmethod
    def _get_word_ngrams_and_length(n, sentences):
        """
        Calculates word n-grams for multiple sentences.

        Args:
          n: wich n-grams to calculate
          sentences: list of string

        Returns:
          A set of n-grams, their frequency and #n-grams in sentences
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        assert len(sentences) > 0
        assert n > 0

        tokens = Rouge._split_into_words(sentences)
        return Rouge._get_ngrams(n, tokens), tokens, len(tokens) - (n - 1)

    @staticmethod
    def _get_unigrams(sentences):
        """
        Calcualtes uni-grams.

        Args:
          sentences: list of string

        Returns:
          A set of n-grams and their frequency
        """
        assert len(sentences) > 0

        tokens = Rouge._split_into_words(sentences)
        unigram_set = collections.defaultdict(int)
        for token in tokens:
            unigram_set[token] += 1
        return unigram_set, len(tokens)

    @staticmethod
    def _get_word_skip_bigrams_and_length(sentences, use_u, max_skip_bigram):
        """
        Calculates word skip bigrams
        :param sentences: List of strings.
        :param max_skip_bigram: The maximum allowed gap between two words in a skip bigram.
        :return: A set of skip bigrams and their frequency
        """
        assert len(sentences) > 0

        tokens = Rouge._split_into_words(sentences)
        if use_u:
            tokens.insert(0, '@start_of_sentence@')
        skip_bigram_set = collections.defaultdict(int)
        for i in range(len(tokens) - 1):
            for j in range(i + 1, min(len(tokens), i + max_skip_bigram + 2)):
                skip_bigram_set[(tokens[i], tokens[j])] += 1
        return skip_bigram_set, ((len(tokens) - (max_skip_bigram + 1))*(max_skip_bigram + 1)) + ((max_skip_bigram + 1) * max_skip_bigram / 2)

    @staticmethod
    def _compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=0.5, weight_factor=1.0):
        """
        Compute precision, recall and f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          evaluated_count: #n-grams in the hypothesis
          reference_count: #n-grams in the reference
          overlapping_count: #n-grams in common between hypothesis and reference
          alpha: Value to use for the F1 score (default: 0.5)
          weight_factor: Weight factor if we have use ROUGE-W (default: 1.0, no impact)

        Returns:
          A dict with 'p', 'r' and 'f' as keys fore precision, recall, f1 score
        """
        precision = 0.0 if evaluated_count == 0 else overlapping_count / float(evaluated_count)
        if weight_factor != 1.0:
            precision = precision ** (1.0 / weight_factor)
        recall = 0.0 if reference_count == 0 else overlapping_count / float(reference_count)
        if weight_factor != 1.0:
            recall = recall ** (1.0 / weight_factor)
        f1_score = Rouge._compute_f_score(precision, recall, alpha)
        return {"f": f1_score, "p": precision, "r": recall}

    @staticmethod
    def _compute_f_score(precision, recall, alpha=0.5):
        """
        Compute f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          precision: precision
          recall: recall
          overlapping_count: #n-grams in common between hypothesis and reference

        Returns:
            f1 score
        """
        return 0.0 if (recall == 0.0 or precision == 0.0) else precision * recall / ((1 - alpha) * precision + alpha * recall)

    @staticmethod
    def _compute_ngrams(evaluated_sentences, reference_sentences, n):
        """
        Computes n-grams overlap of two text collections of sentences.
        Source: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf

        Args:
          evaluated_sentences: The sentences that have been picked by the
                               summarizer
          reference_sentences: The sentences from the referene set
          n: Size of ngram

        Returns:
          Number of n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times

        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        # Modified from https://github.com/pltrdy/seq2seq/blob/master/seq2seq/metrics/rouge.py
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams, _, evaluated_count = Rouge._get_word_ngrams_and_length(n, evaluated_sentences)
        reference_ngrams, _, reference_count = Rouge._get_word_ngrams_and_length(n, reference_sentences)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = set(evaluated_ngrams.keys()).intersection(set(reference_ngrams.keys()))
        overlapping_count = 0
        for ngram in overlapping_ngrams:
            overlapping_count += min(evaluated_ngrams[ngram], reference_ngrams[ngram])

        return evaluated_count, reference_count, overlapping_count

    @staticmethod
    def _compute_ngrams_lcs(evaluated_sentences, reference_sentences, weight_factor=1.0):
        """
        Computes ROUGE-L (summary level) of two text collections of sentences.
        http://research.microsoft.com/en-us/um/people/cyl/download/papers/
        rouge-working-note-v1.3.1.pdf
        Args:
          evaluated_sentences: The sentences that have been picked by the summarizer
          reference_sentence: One of the sentences in the reference summaries
          weight_factor: Weight factor to be used for WLCS (1.0 by default if LCS)
        Returns:
          Number of LCS n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times
        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        def _lcs(x, y):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(int)
            dirs = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        vals[i, j] = vals[i - 1, j - 1] + 1
                        dirs[i, j] = '|'
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = '^'
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = '<'

            return vals, dirs

        def _wlcs(x, y, weight_factor):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(float)
            dirs = collections.defaultdict(int)
            lengths = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        length_tmp = lengths[i - 1, j - 1]
                        vals[i, j] = vals[i - 1, j - 1] + (length_tmp + 1) ** weight_factor - length_tmp ** weight_factor
                        dirs[i, j] = '|'
                        lengths[i, j] = length_tmp + 1
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = '^'
                        lengths[i, j] = 0
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = '<'
                        lengths[i, j] = 0

            return vals, dirs

        def _mark_lcs(mask, dirs, m, n):
            while m != 0 and n != 0:
                if dirs[m, n] == '|':
                    m -= 1
                    n -= 1
                    mask[m] = 1
                elif dirs[m, n] == '^':
                    m -= 1
                elif dirs[m, n] == '<':
                    n -= 1
                else:
                    raise UnboundLocalError('Illegal move')

            return mask

        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_unigrams_dict, evaluated_count = Rouge._get_unigrams(evaluated_sentences)
        reference_unigrams_dict, reference_count = Rouge._get_unigrams(reference_sentences)

        # Has to use weight factor for WLCS
        use_WLCS = weight_factor != 1.0
        if use_WLCS:
            evaluated_count = evaluated_count ** weight_factor
            reference_count = 0

        overlapping_count = 0.0
        for reference_sentence in reference_sentences:
            reference_sentence_tokens = reference_sentence.split()
            if use_WLCS:
                reference_count += len(reference_sentence_tokens) ** weight_factor
            hit_mask = [0 for _ in range(len(reference_sentence_tokens))]

            for evaluated_sentence in evaluated_sentences:
                evaluated_sentence_tokens = evaluated_sentence.split()

                if use_WLCS:
                    _, lcs_dirs = _wlcs(reference_sentence_tokens, evaluated_sentence_tokens, weight_factor)
                else:
                    _, lcs_dirs = _lcs(reference_sentence_tokens, evaluated_sentence_tokens)
                _mark_lcs(hit_mask, lcs_dirs, len(reference_sentence_tokens), len(evaluated_sentence_tokens))

            overlapping_count_length = 0
            for ref_token_id, val in enumerate(hit_mask):
                if val == 1:
                    token = reference_sentence_tokens[ref_token_id]
                    if evaluated_unigrams_dict[token] > 0 and reference_unigrams_dict[token] > 0:
                        evaluated_unigrams_dict[token] -= 1
                        reference_unigrams_dict[ref_token_id] -= 1

                        if use_WLCS:
                            overlapping_count_length += 1
                            if (ref_token_id + 1 < len(hit_mask) and hit_mask[ref_token_id + 1] == 0) or ref_token_id + 1 == len(hit_mask):
                                overlapping_count += overlapping_count_length ** weight_factor
                                overlapping_count_length = 0
                        else:
                            overlapping_count += 1

        if use_WLCS:
            reference_count = reference_count ** weight_factor

        return evaluated_count, reference_count, overlapping_count

    @staticmethod
    def _compute_skip_bigrams(evaluated_sentences, reference_sentences, use_u, max_skip_bigram):
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_skip_bigrams_dict, evaluated_count = Rouge._get_word_skip_bigrams_and_length(
            sentences=evaluated_sentences,
            use_u=use_u,
            max_skip_bigram=max_skip_bigram
        )
        reference_skip_bigrams_dict, reference_count = Rouge._get_word_skip_bigrams_and_length(
            sentences=reference_sentences,
            use_u=use_u,
            max_skip_bigram=max_skip_bigram
        )

        overlapping_skip_bigrams = set(evaluated_skip_bigrams_dict.keys()).intersection(set(reference_skip_bigrams_dict.keys()))
        overlapping_count = 0
        for skip_bigram in overlapping_skip_bigrams:
            overlapping_count += min(evaluated_skip_bigrams_dict[skip_bigram], reference_skip_bigrams_dict[skip_bigram])

        return evaluated_count, reference_count, overlapping_count

    def get_scores(self, hypothesis, references):
        """
        Compute precision, recall and f1 score between hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between hypothesis and references

        Raises:
          ValueError: raises exception if a type of hypothesis is different than the one of reference
          ValueError: raises exception if a len of hypothesis is different than the one of reference
        """
        if isinstance(hypothesis, str):
            hypothesis, references = [hypothesis], [references]

        if type(hypothesis) != type(references):
            raise ValueError("'hyps' and 'refs' are not of the same type")

        if len(hypothesis) != len(references):
            raise ValueError("'hyps' and 'refs' do not have the same length")
        scores = {}
        has_rouge_n_metric = False
        has_rouge_l_metric = False
        has_rouge_w_metric = False
        has_rouge_s_metric = False
        has_rouge_su_metric = False
        for metric in self.metrics:
            metric_type = metric.split('-')[-1]
            if metric_type.isdigit():
                has_rouge_n_metric = True
            elif metric_type == 'l':
                has_rouge_l_metric = True
            elif metric_type == 'w':
                has_rouge_w_metric = True
            elif 'su' in metric_type:
                has_rouge_su_metric = True
            else:
                has_rouge_s_metric = True

        if has_rouge_n_metric:
            scores.update(self._get_scores_rouge_n(hypothesis, references))
            # scores = {**scores, **self._get_scores_rouge_n(hypothesis, references)}

        if has_rouge_l_metric:
            scores.update(self._get_scores_rouge_l_or_w(hypothesis, references, False))
            # scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, False)}

        if has_rouge_w_metric:
            scores.update(self._get_scores_rouge_l_or_w(hypothesis, references, True))
            # scores = {**scores, **self._get_scores_rouge_l_or_w(hypothesis, references, True)}

        if has_rouge_s_metric:
            scores.update(self._get_scores_rouge_s(hypothesis, references, use_u=False))

        if has_rouge_su_metric:
            scores.update(self._get_scores_rouge_s(hypothesis, references, use_u=True))

        return scores

    def _get_scores_rouge_n(self, all_hypothesis, all_references):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
        metrics = [metric for metric in self.metrics if metric.split('-')[-1].isdigit()]

        if self.apply_avg or self.apply_best:
            scores = {metric: {stat:0.0 for stat in Rouge.STATS} for metric in metrics}
        else:
            scores = {metric: [{stat:[] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))] for metric in metrics}

        for sample_id, (hypothesis, references) in enumerate(zip(all_hypothesis, all_references)):
            assert isinstance(hypothesis, str)
            has_multiple_references = False
            if isinstance(references, list):
                has_multiple_references = len(references) > 1
                if not has_multiple_references:
                    references = references[0]

            # Prepare hypothesis and reference(s)
            hypothesis = self._preprocess_summary_as_a_whole(hypothesis)
            references = [self._preprocess_summary_as_a_whole(reference) for reference in references] if has_multiple_references else [self._preprocess_summary_as_a_whole(references)]

            # Compute scores
            for metric in metrics:
                suffix = metric.split('-')[-1]
                n = int(suffix)

                # Aggregate
                if self.apply_avg:
                    # average model
                    total_hypothesis_ngrams_count = 0
                    total_reference_ngrams_count = 0
                    total_ngrams_overlapping_count = 0

                    for reference in references:
                        hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                        total_hypothesis_ngrams_count += hypothesis_count
                        total_reference_ngrams_count += reference_count
                        total_ngrams_overlapping_count += overlapping_ngrams

                    score = Rouge._compute_p_r_f_score(total_hypothesis_ngrams_count, total_reference_ngrams_count, total_ngrams_overlapping_count, self.alpha)

                    for stat in Rouge.STATS:
                        scores[metric][stat] += score[stat]
                else:
                    # Best model
                    if self.apply_best:
                        best_current_score = None
                        for reference in references:
                            hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                            score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                            if best_current_score is None or score['r'] > best_current_score['r']:
                                best_current_score = score

                        for stat in Rouge.STATS:
                            scores[metric][stat] += best_current_score[stat]
                    # Keep all
                    else:
                        for reference in references:
                            hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams(hypothesis, reference, n)
                            score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                            for stat in Rouge.STATS:
                                scores[metric][sample_id][stat].append(score[stat])

        # Compute final score with the average or the the max
        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for metric in metrics:
                for stat in Rouge.STATS:
                    scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _get_scores_rouge_l_or_w(self, all_hypothesis, all_references, use_w=False):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        Args:
          hypothesis: hypothesis summary, string
          references: reference summary/ies, either string or list of strings (if multiple)

        Returns:
          Return precision, recall and f1 score between all hypothesis and references
        """
        metric = "rouge-w" if use_w else "rouge-l"
        if self.apply_avg or self.apply_best:
            scores = {metric: {stat:0.0 for stat in Rouge.STATS}}
        else:
            scores = {metric: [{stat:[] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))]}

        for sample_id, (hypothesis_sentences, references_sentences) in enumerate(zip(all_hypothesis, all_references)):
            assert isinstance(hypothesis_sentences, str)
            has_multiple_references = False
            if isinstance(references_sentences, list):
                has_multiple_references = len(references_sentences) > 1
                if not has_multiple_references:
                    references_sentences = references_sentences[0]

            # Prepare hypothesis and reference(s)
            hypothesis_sentences = self._preprocess_summary_per_sentence(hypothesis_sentences)
            references_sentences = [self._preprocess_summary_per_sentence(reference) for reference in references_sentences] if has_multiple_references else [self._preprocess_summary_per_sentence(references_sentences)]

            # Compute scores
            # Aggregate
            if self.apply_avg:
                # average model
                total_hypothesis_ngrams_count = 0
                total_reference_ngrams_count = 0
                total_ngrams_overlapping_count = 0

                for reference_sentences in references_sentences:
                    hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0)
                    total_hypothesis_ngrams_count += hypothesis_count
                    total_reference_ngrams_count += reference_count
                    total_ngrams_overlapping_count += overlapping_ngrams

                score = Rouge._compute_p_r_f_score(total_hypothesis_ngrams_count, total_reference_ngrams_count, total_ngrams_overlapping_count, self.alpha, self.weight_factor)

                for stat in Rouge.STATS:
                    scores[metric][stat] += score[stat]
            else:
                # Best model
                if self.apply_best:
                    best_current_score = None
                    best_current_score_wlcs = None
                    for reference_sentences in references_sentences:
                        hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0)
                        score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor)

                        if use_w:
                            reference_count_for_score = reference_count ** (1.0 / self.weight_factor)
                            overlapping_ngrams_for_score = overlapping_ngrams
                            score_wlcs = (overlapping_ngrams_for_score / reference_count_for_score) ** (1.0 / self.weight_factor)

                            if best_current_score_wlcs is None or score_wlcs > best_current_score_wlcs:
                                best_current_score = score
                                best_current_score_wlcs = score_wlcs
                        else:
                            if best_current_score is None or score['r'] > best_current_score['r']:
                                best_current_score = score

                    for stat in Rouge.STATS:
                        scores[metric][stat] += best_current_score[stat]
                # Keep all
                else:
                    for reference_sentences in references_sentences:
                        hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_ngrams_lcs(hypothesis_sentences, reference_sentences, self.weight_factor if use_w else 1.0)
                        score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha, self.weight_factor)

                        for stat in Rouge.STATS:
                            scores[metric][sample_id][stat].append(score[stat])

        # Compute final score with the average or the the max
        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for stat in Rouge.STATS:
                scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _get_scores_rouge_s(self, all_hypothesis, all_references, use_u=False):
        """
        Computes precision, recall and f1 score between all hypothesis and references

        :param all_hypothesis: A list of strings containing system generated summaries.
        :param all_references:
            If multiple references: A list of lists of strings containing human generated summaries.
            If single reference: A list of strings containing human generated summaries.
        :param use_u: If true ROUGE-SU is calculated
        :return:
        """
        metric = "rouge-su{}".format(self.max_skip_bigram) if use_u else "rouge-s{}".format(self.max_skip_bigram)
        if self.apply_avg or self.apply_best:
            scores = {metric: {stat:0.0 for stat in Rouge.STATS}}
        else:
            scores = {metric: [{stat:[] for stat in Rouge.STATS} for _ in range(len(all_hypothesis))]}

        for sample_id, (hypothesis, references) in enumerate(zip(all_hypothesis, all_references)):
            assert isinstance(hypothesis, str)
            has_multiple_references = False
            if isinstance(references, list):
                has_multiple_references = len(references) > 1
                if not has_multiple_references:
                    references = references[0]

            # Prepare hypothesis and reference(s)
            hypothesis = self._preprocess_summary_as_a_whole(hypothesis)
            references = [self._preprocess_summary_as_a_whole(reference) for reference in references] if has_multiple_references else [self._preprocess_summary_as_a_whole(references)]

            # Compute scores
            # Aggregate
            if self.apply_avg:
                # average model
                total_hypothesis_skip_bigrams_count = 0
                total_reference_skip_bigrams_count = 0
                total_skip_bigrams_overlapping_count = 0

                for reference in references:
                    hypothesis_count, reference_count, overlapping_skip_bigrams = Rouge._compute_skip_bigrams(
                        evaluated_sentences=hypothesis,
                        reference_sentences=reference,
                        use_u=use_u,
                        max_skip_bigram=self.max_skip_bigram
                    )
                    total_hypothesis_skip_bigrams_count += hypothesis_count
                    total_reference_skip_bigrams_count += reference_count
                    total_skip_bigrams_overlapping_count += overlapping_skip_bigrams

                score = Rouge._compute_p_r_f_score(total_hypothesis_skip_bigrams_count, total_reference_skip_bigrams_count, total_skip_bigrams_overlapping_count, self.alpha)

                for stat in Rouge.STATS:
                    scores[metric][stat] += score[stat]
            elif self.apply_best:
                best_current_score = None
                for reference in references:
                    hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_skip_bigrams(
                        evaluated_sentences=hypothesis,
                        reference_sentences=reference,
                        use_u=use_u,
                        max_skip_bigram=self.max_skip_bigram
                    )
                    score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                    if best_current_score is None or score['r'] > best_current_score['r']:
                        best_current_score = score

                for stat in Rouge.STATS:
                    scores[metric][stat] += best_current_score[stat]
            else: # Keep all
                for reference in references:
                    hypothesis_count, reference_count, overlapping_ngrams = Rouge._compute_skip_bigrams(
                        evaluated_sentences=hypothesis,
                        reference_sentences=reference,
                        use_u=use_u,
                        max_skip_bigram=self.max_skip_bigram
                    )
                    score = Rouge._compute_p_r_f_score(hypothesis_count, reference_count, overlapping_ngrams, self.alpha)
                    for stat in Rouge.STATS:
                        scores[metric][sample_id][stat].append(score[stat])

        # Compute final score with the average or the the max
        if (self.apply_avg or self.apply_best) and len(all_hypothesis) > 1:
            for stat in Rouge.STATS:
                scores[metric][stat] /= len(all_hypothesis)

        return scores

    def _preprocess_summary_as_a_whole(self, summary):
        """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, remove stopwords if enable, lowering) of a summary as a whole

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        # Truncate
        if self.limit_length:
            # By words
            if self.length_limit_type == 'words':
                summary = ' '.join(sentences)
                all_tokens = summary.split() # Counting as in the perls script
                summary = ' '.join(all_tokens[:self.length_limit])

            # By bytes
            elif self.length_limit_type == 'bytes':
                summary = ''
                current_len = 0
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)

                    if current_len + sentence_len < self.length_limit:
                        if current_len != 0:
                            summary += ' '
                        summary += sentence
                        current_len += sentence_len
                    else:
                        if current_len > 0:
                            summary += ' '
                        summary += sentence[:self.length_limit-current_len]
                        break
        else:
            summary = ' '.join(sentences)

        summary = Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary.lower()).strip()

        # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot" and "can not" as "can not",
        # we have to hack nltk tokenizer to not transform "cannot/can not" to "can not"
        if self.ensure_compatibility:
            tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', summary))
        else:
            tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', summary))

        if self.stopword_removal:
            self.remove_stopwords(tokens) # stopword removal in place

        if self.stemming:
            self.stem_tokens(tokens) # stemming in-place

        if self.ensure_compatibility:
            preprocessed_summary = [Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))]
        else:
            preprocessed_summary = [' '.join(tokens)]

        return preprocessed_summary

    def _preprocess_summary_per_sentence(self, summary):
        """
        Preprocessing (truncate text if enable, tokenization, stemming if enable, lowering) of a summary by sentences

        Args:
          summary: string of the summary

        Returns:
          Return the preprocessed summary (string)
        """
        sentences = Rouge.split_into_sentences(summary, self.ensure_compatibility)

        # Truncate
        if self.limit_length:
            final_sentences = []
            current_len = 0
            # By words
            if self.length_limit_type == 'words':
                for sentence in sentences:
                    tokens = sentence.strip().split()
                    tokens_len = len(tokens)
                    if current_len + tokens_len < self.length_limit:
                        sentence = ' '.join(tokens)
                        final_sentences.append(sentence)
                        current_len += tokens_len
                    else:
                        sentence = ' '.join(tokens[:self.length_limit - current_len])
                        final_sentences.append(sentence)
                        break
            # By bytes
            elif self.length_limit_type == 'bytes':
                for sentence in sentences:
                    sentence = sentence.strip()
                    sentence_len = len(sentence)
                    if current_len + sentence_len < self.length_limit:
                        final_sentences.append(sentence)
                        current_len += sentence_len
                    else:
                        sentence = sentence[:self.length_limit - current_len]
                        final_sentences.append(sentence)
                        break
            sentences = final_sentences

        final_sentences = []
        for sentence in sentences:
            sentence = Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence.lower()).strip()

            # Preprocess. Hack: because official ROUGE script bring "cannot" as "cannot" and "can not" as "can not",
            # we have to hack nltk tokenizer to not transform "cannot/can not" to "can not"
            if self.ensure_compatibility:
                tokens = self.tokenize_text(Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', sentence))
            else:
                tokens = self.tokenize_text(Rouge.REMOVE_CHAR_PATTERN.sub(' ', sentence))

            if self.stemming:
                self.stem_tokens(tokens) # stemming in-place

            if self.ensure_compatibility:
                sentence = Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
            else:
                sentence = ' '.join(tokens)

            final_sentences.append(sentence)

        return final_sentences
