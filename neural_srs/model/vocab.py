import sqlite3
import requests
import re


DECK_IDS = [1604047906179, 1533367265407]


def strip_tags(text):
    return re.sub('<[^<]+?>', '', text)


def get_vocab_in_decks():
    results = set()
    conn = sqlite3.connect('../../collection.anki2')
    c = conn.cursor()
    c.execute('''
        select flds from cards join notes on cards.nid = notes.id where did=1604047906179 or did=1533367265407;
    ''')
    for item in c.fetchall():
        fields = item[0]
        vocab = strip_tags(fields.split('\x1f')[0])
        results.add(vocab)
    return results


def load_spreadsheet_data():
    all_vocab = []
    rank_lookup = {}
    with open('/home/brendan/Documents/vocab.csv') as f:
        for i, row in enumerate(f):
            if i == 0 or i == 1:
                continue
            vocab = row.split(',')[0]
            all_vocab.append(vocab)
            rank_lookup[vocab] = i + 1
    return all_vocab, rank_lookup


def load_netflix_data():
    with open('./Netflix10K.txt') as f:
        return f.read().split('\n')


def make_data():
    added_words = set()
    vocab_in_decks = get_vocab_in_decks()
    spreadsheet_vocab, rank_lookup = load_spreadsheet_data()
    netflix_vocab = load_netflix_data()

    with open('to_add2.csv', 'w+') as f:
        i = 0
        for rank, word in enumerate(spreadsheet_vocab[1000:10000]):
            if word not in vocab_in_decks:
                definition, reading, count = get_definition(word)
                if definition is None:
                    continue
                added_words.add(word)
                print(f"Rank: {rank} Adding {word}, definition: {definition}")
                f.write(f'{word}|{reading}|{rank}|SPREADSHEET|{definition}|{count}\n')
            i += 1
            if i % 10 == 0:
                f.flush()

        for rank, word in enumerate(netflix_vocab[:4000]):
            if word not in vocab_in_decks and word not in added_words:
                definition, reading, count = get_definition(word)
                if definition is None:
                    continue
                print(f"Rank: {rank} Adding {word}, definition: {definition}")
                f.write(f'{word}|{reading}|{rank}|NETFLIX|{definition}|{count}\n')
            i += 1
            if i % 10 == 0:
                f.flush()


def get_definition(word):
    response = requests.get(f"https://jisho.org/api/v1/search/words?keyword={word}")
    results = response.json()
    data = results.get('data')[0] if results.get('data') is not None and len(results.get('data')) > 0 else None
    if data is None:
        return None, None, 0

    reading_lookup = {}
    for japanese in data.get('japanese'):
        if japanese.get('word') is not None and japanese['word'] not in reading_lookup:
            reading_lookup[japanese['word']] = japanese.get('reading')
    all_words = [japanese.get('word') for japanese in data.get('japanese') if japanese.get('word') is not None]
    all_readings = [japanese.get('reading') for japanese in data.get('japanese') if japanese.get('reading') is not None]

    senses = data.get('senses')
    number_of_definitions = len(senses)
    if len(senses) == 0:
        return None, None, 0

    definition = ';'.join(senses[0].get('english_definitions'))
    if word in all_words or word in all_readings:
        if word in reading_lookup:
            reading = reading_lookup[word]
        else:
            reading = word
        return definition, reading, number_of_definitions
    return None, None, 0


def get_overlap_single(target_words, base_words, filename):
    target_words_set = set(target_words[:10000])
    big_target_words_set = set(target_words)
    base_words = base_words[:10000]

    with open(filename, 'w+') as f:
        for i, word in enumerate(base_words):
            if word not in big_target_words_set:
                f.write(f'{i},{word},NO\n')
            elif word in target_words_set:
                f.write(f'{i},{word},,,YES\n')
            else:
                f.write(f'{i},{word},,BIG\n')


def get_overlap():
    spreadsheet_vocab, _ = load_spreadsheet_data()
    netflix_vocab = load_netflix_data()

    get_overlap_single(spreadsheet_vocab, netflix_vocab, 'netflix_base.csv')
    get_overlap_single(netflix_vocab, spreadsheet_vocab, 'spreadsheet_base.csv')


def main():
    make_data()


if __name__ == '__main__':
    main()


