from keras.preprocessing.text import Tokenizer, maketrans
import string


# a tokenizer that supports out of vocabulary tokens
class OOVTokenizer(Tokenizer):
    def base_filter(self):
        f = string.punctuation
        f = f.replace("'", '')
        f += '\t\n'
        return f

    # this is added to the class to support overriding in subclasses, like the code tokenizer
    def text_to_word_sequence(self, text, filters=base_filter(), lower=True, split=" "):
        '''prune: sequence of characters to filter out
        '''
        if lower:
            text = text.lower()
        text = text.translate(maketrans(filters, split*len(filters)))
        seq = text.split(split)
        return [_f for _f in seq if _f]

    def fit_on_texts(self, texts):
        '''Required before using texts_to_sequences or texts_to_matrix

        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        '''
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = text if self.char_level else self.text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.oov_token = len(sorted_voc) + 1

        self.index_docs = {}
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences_generator(self, texts):
        '''Transforms each text in texts in a sequence of integers.
        Only top "nb_words" most frequent words will be taken into account.
        Words that are not known by the tokenizer will be replaced by self.oov_token token.

        Yields individual sequences.

        # Arguments:
            texts: list of strings.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text if self.char_level else self.text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        vect.append(self.oov_token)
                    else:
                        vect.append(i)
                else:
                    vect.append(self.oov_token)
            yield vect