from keras.preprocessing.text import Tokenizer, maketrans
from oov_tokenizer import OOVTokenizer
# we do not want to filter anything in code.
def base_code_filter():
    return ""



# this class overrides methods to be able to use the text_to_word_sequence function above.
class CodeTokenizer(OOVTokenizer):
    def text_to_word_sequence(self, text, filters=base_code_filter(), lower=True, split=" "):
        ''' Tokenizes code. All consecutive alphanumeric characters are grouped into one token.
        Thereby trying to heuristically match identifiers.
        All other symbols are seen as one token.
        Whitespace is stripped, except the newline token.
        '''
        if lower:
            text = text.lower() #type: str
        seq = []
        curr = ""
        for c in text:
            if c.isalnum():
                curr += c
            else:
                if curr != "":
                    seq.append(curr)
                    curr = ""
                if not c.isspace() or c == '\n':
                    seq.append(c)
        return [_f for _f in seq if _f]
