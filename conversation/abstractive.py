open_quote = ['“']
close_quote = ['”']
said_words = ['said', 'mentioned']
reply_words = ['replied', 'responded']
transitions = ['so', 'also', 'and', 'well', 'ok', 'okay']

def flatten(nested_list):
    return [label for thread in nested_list for label in thread]

def change_tense(word):
    first_tense_dict = {
        'i': 'they',
        'i\'m': 'they\'re',
        'i\'ve': 'they\'ve',
        'i\'d': 'they\'d',
        'am': 'are',
        'me': 'them',
        'my': 'their',
        'mine': 'theirs'
    }
    if word.lower() in first_tense_dict:
        return first_tense_dict[word.lower()]
    else:
        return word

def generate_formatted(sentences, authors):
    formatted_result = []

    for sentence_index, sentence in enumerate(sentences):
        sentence_tokens = sentence.split()
        processed = ''
        in_quotes = False
        for token in sentence_tokens:
            # Don't fiddle with tense of words that are quoted
            if token[0] in open_quote:
                in_quotes = True
            if token[-1] in close_quote:
                in_quotes = False

            if in_quotes:
                processed += token + ' '
            else:
                processed += change_tense(token) + ' '
        
        formatted_sentence = authors[sentence_index].strip() + ' said ' + processed.strip()
        formatted_result.append(formatted_sentence)

    print(formatted_result)
    return formatted_result

def process_files():
    with open('extractive.txt', 'r') as extractive, open('authors.txt', 'r') as authors, open('abstractive.txt', 'w') as abstractive:
        author_lines = authors.readlines()
        extractive_lines = extractive.readlines()

        formatted = '\n'.join(generate_formatted(extractive_lines, author_lines))

        abstractive.write(formatted + '\n')

# If this class is being run, process the files manually.
if __name__ == '__main__':
    process_files()