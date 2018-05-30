
def change_tense(word):
    first_tense_dict = {
        'i': 'they',
        'i\'m': 'they\'re',
        'i\'ve': 'they\'ve',
        'am': 'are',
        'me': 'them',
        'my': 'their',
        'mine': 'theirs'
    }
    if word.lower() in first_tense_dict:
        return first_tense_dict[word.lower()]
    else:
        return word

with open('extractive.txt', 'r') as extractive, open('authors.txt', 'r') as authors, open('abstractive.txt', 'w') as abstractive:
    author_lines = authors.readlines()
    open_quote = ['“']
    close_quote = ['”']
    said_words = ['said', 'mentioned']
    reply_words = ['replied', 'responded']
    transitions = ['so', 'also', 'and', 'well', 'ok', 'okay']
    for line_index, line in enumerate(extractive):
        line_tokens = line.split()
        processed_line = ''
        in_quotes = 0
        for token in line_tokens:
            # Don't change tense of words that are quoted
            if token[0] in open_quote:
                in_quotes = 1
            if token[-1] in close_quote:
                in_quotes = 0

            if in_quotes:
                processed_line += token + ' '
            else:
                processed_line += change_tense(token) + ' '

        abstr_str = author_lines[line_index].rstrip() + ' said ' + processed_line.strip()
        abstractive.write(abstr_str + '\n')
