from parsers.ChatParser import ChatParser
from nltk.tokenize import TextTilingTokenizer
import pdb

parser = ChatParser('../data/gnue/tiny', False)
tokenizer = TextTilingTokenizer()

data = parser.parse('train')
text = '\n\n'.join(data['data'][0][0])

tokens = tokenizer.tokenize(text)

pdb.set_trace()