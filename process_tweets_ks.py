from gensim import corpora
import re
import argparse
import twokenize
import codecs

URL_PATTERN = re.compile(ur'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
USER_PATTERN = re.compile(ur'@\w+')
NUM_PATTERN = re.compile(ur'[0-9]+')
PUNCT_PATTERN = re.compile(ur'\W{4,}')
PATTERNS = {
    'URL': URL_PATTERN,
    'USER': USER_PATTERN,
    'NUM': NUM_PATTERN
}


class Line(object):
    def __init__(self, raw, line_no):
        parts = raw.split(',')
        self.line_no = line_no
        self.tid = None
        self.txt = None
        self.label = None
        self.error = False
        if len(parts) != 6:
            bad_parts = parts[5:]
            new_str = ', '.join(bad_parts)
            parts = parts[:5] + [new_str]
        if len(parts) != 6:
            print('parsing: bad line @ %d (parts len = %d): %s' % (self.line_no, len(parts), raw))
            # print(parts)
            self.error = True
        else:
            self.tid = parts[1].strip('"')
            self.txt = parts[5].strip('"')
            self.label = int(parts[0].strip('"'))
        self.tokens = None

    def __repr__(self):
        if self.txt is not None:
            return 'Line(%s)' % self.txt
        else:
            return 'Line(ERROR)'


def tokenize_lines(lines, queue):
    results = []
    for line in lines:
        tokens = tokenize(line)
        if tokens is not None:
            results.append(tokens)
    queue.put(results)


def tokenize(line):
    text = line.txt
    if text is not None:
        text = text.lower()
        text = re.sub(URL_PATTERN, 'URL', text)
        text = re.sub(USER_PATTERN, 'USER', text)
        text = re.sub(PUNCT_PATTERN, ' ', text)
        tokens = twokenize.tokenizeRawTweetText(text)
        line.tokens = tokens
        return tokens
    else:
        return None


def process(filename, dictionary=None):
    unicode_errors = 0
    parse_errors = 0
    lines = []
    count = 0
    with codecs.open(filename, 'rU', 'utf-8', 'ignore') as f:
        for line in f:
            parsed = Line(line.rstrip('\n'), count)
            if parsed.error:
                parse_errors += 1
            else:
                lines.append(parsed)
            count += 1
    print('%d / %d lines with unicode errors' % (unicode_errors, count))
    print('%d / %d lines with parse errors' % (parse_errors, count))

    make_dict = False
    if dictionary is None:
        dictionary = corpora.Dictionary()
        make_dict = True

    nlines = len(lines)
    ncomplete = 0
    processed_lines = []
    for line in lines:
        tokens = tokenize(line)
        if tokens is not None:
            processed_lines.append(line)
            if make_dict:
                dictionary.add_documents([tokens])
            ncomplete += 1
        if (ncomplete % 10000) == 0:
            print('%d / %d' % (ncomplete, nlines))

    if make_dict:
        dictionary.filter_extremes(no_below=args.filter_below, no_above=args.filter_above, keep_n=None)
        # remove stop words
        stoplist = []
        with open(args.stop_words, 'r') as f:
            stoplist = map(lambda x: x.strip().lower(), f.readlines())
        stop_ids = [dictionary.token2id[sw] for sw in stoplist if sw in dictionary.token2id]
        dictionary.filter_tokens(stop_ids)
        # remove gaps
        dictionary.compactify()
        print(dictionary)

    return processed_lines, dictionary


def write(outfile, lines, dictionary):
    dictionary.save_as_text(outfile+".vocab.txt", sort_by_word=False)
    outf = codecs.open(outfile, 'w+', 'utf-8')
    outf_words = codecs.open(outfile + '.words', 'w+', 'utf-8')
    label_map = {0: "negative", 2: "neutral", 4: "positive"}
    nskips = 0
    for line in lines:
        if line.txt is None or line.tokens is None or len(line.tokens) == 0:
            nskips += 1
        else:
            ints = []
            for w in line.tokens:
                if w in dictionary.token2id:
                    ints.append(str(dictionary.token2id[w]))
            text = ' '.join(ints)
            outline = '%s\t%s\t%s\n' % (line.tid, label_map[line.label], text)
            outf.write(outline)
            words = []
            for w in line.tokens:
                if w in dictionary.token2id:
                    words.append(w)
            text = ' '.join(words)
            outline = '%s\t%s\t%s\n' % (line.tid, label_map[line.label], text)
            outf_words.write(outline)
    print('skipped %d lines' % nskips)


def main(args):
    filename = args.tweet_file
    lines, dictionary = process(filename)
    outfile = args.output_file
    write(outfile, lines, dictionary)
    if args.test_file:
        testfile = args.test_file
        print('processing test file %s' % testfile)
        lines, dictionary = process(testfile, dictionary=dictionary)
        write(outfile + '.test', lines, dictionary)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process 1.6M tweet corpus")
    parser.add_argument('--tweet-file', help="location of 1.6M corpus", required=True)
    parser.add_argument('--output-file', help="location, name of output", required=True)
    parser.add_argument('--test-file', help="location of test file", default=None)
    parser.add_argument('--stop-words', help="location of stopwords", default=None)
    parser.add_argument('--V', help="vocab size", default=None)
    parser.add_argument('--filter-below', help='filter words occuring less than this many times', type=int)
    parser.add_argument('--filter-above', help='filter words occurring in this % of documents', type=float)

    args = parser.parse_args()
    print(args)
    main(args)



