#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
__file__

    nlp_utils.py

__description__

    This file provides functions to perform NLP task, e.g., TF-IDF and POS tagging.

__author__

    Lei Xu < leixuast@gmail.com >

"""
import re
import sys
import nltk
from bs4 import BeautifulSoup
from replacer import CsvWordReplacer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
sys.path.append("../")
from param_config import config
from textacy.preprocess import (remove_accents, 
                                fix_bad_unicode, transliterate_unicode,normalize_whitespace,
                                )
import ftfy
reload(sys)
sys.setdefaultencoding('utf-8')

################
## Stop Words ##
################
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)


##############
## Stemming ##
##############
if config.stemmer_type == "porter":
    english_stemmer = nltk.stem.PorterStemmer()
elif config.stemmer_type == "snowball":
    english_stemmer = nltk.stem.SnowballStemmer('english')
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed
	

#############
## POS Tag ##
#############
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
def pos_tag_text(line,
                 token_pattern=token_pattern,
                 exclude_stopword=config.cooccurrence_word_exclude_stopword,
                 encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    for name in ["query", "product_title", "product_description"]:
        l = line[name]
        ## tokenize
        tokens = [x.lower() for x in token_pattern.findall(l)]
		## stem
        tokens = stem_tokens(tokens, english_stemmer)
        if exclude_stopword:
            tokens = [x for x in tokens if x not in stopwords]
        tags = pos_tag(tokens)
        tags_list = [t for w,t in tags]
        tags_str = " ".join(tags_list)
        #print tags_str
        line[name] = tags_str
    return line

    
############
## TF-IDF ##
############
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
   
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
tfidf__norm = "l2"
tfidf__max_df = 1.0

tfidf__min_df = 1

def getTFV(token_pattern = token_pattern,
           norm = tfidf__norm,
           max_df = tfidf__max_df,
           min_df = tfidf__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words = stop_words, norm=norm, vocabulary=vocabulary)
    return tfv
   

#########
## BOW ##
#########
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
   
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
bow__max_df = 0.75
bow__min_df = 3
def getBOW(token_pattern = token_pattern,
           max_df = bow__max_df,
           min_df = bow__min_df,
           ngram_range = (1, 1),
           vocabulary = None,
           stop_words = 'english'):
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None, 
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words = stop_words, vocabulary=vocabulary)
    return bow




#adapted from http://textacy.readthedocs.io/en/latest/_modules/textacy/preprocess.html#replace_numbers
def unpack_contractions(text):
    """
    Replace *English* contractions in ``text`` str with their unshortened forms.
    N.B. The "'d" and "'s" forms are ambiguous (had/would, is/has/possessive),
    so are left as-is.
    """
    # standard
    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou|[Ww]here|[Ww]hen|[Hh]ow)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
    # non-standard
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)
 


    return text




################
## Text Clean ##
################
## synonym replacer
replacer = CsvWordReplacer('%s/synonyms.csv' % config.data_folder)
## other replace dict
## such dict is found by exploring the training data

CURRENCIES_LONG = {'$': 'dollar', 'zł': 'zloty', '£': 'pound', '¥': 'yuan', '฿': 'baht',
              '₡': 'colon', '₦': 'naira', '₩': 'won', '₪': 'shekel', '₫': 'dong',
              '€': 'euro', '₱': 'peso', '₲': 'guarani', '₴': 'hryvnia', '₹': 'rupee'
                   }



CURRENCIES = {'$': 'USD', 'zł': 'PLN', '£': 'GBP', '¥': 'JPY', '฿': 'THB',
              '₡': 'CRC', '₦': 'NGN', '₩': 'KRW', '₪': 'ILS', '₫': 'VND',
              '€': 'EUR', '₱': 'PHP', '₲': 'PYG', '₴': 'UAH', '₹': 'INR'}

CURRENCIES_abbr = {'USD': 'dollar', 'PLN': 'zloty', 'GBP': 'pound', 'JPY': 'yuan', 'THB': 'baht',
              'CRC': 'colon', 'NGN': 'naira', 'KRW': 'won', 'ILS': 'shekel', 'VND': 'dong',
              'EUR': 'euro', 'PHP': 'peso', 'PYG': 'guarani', 'UAH': 'hryvnia', 'INR': 'rupee'
                   }


def replace_currency_symbols(text, replace_with=None):
    """
    Replace all currency symbols in ``text`` str with string specified by ``replace_with`` str.

    Args:
        text (str): raw text
        replace_with (str): if None (default), replace symbols with
            their standard 3-letter abbreviations (e.g. '$' with 'USD', '£' with 'GBP');
            otherwise, pass in a string with which to replace all symbols
            (e.g. "*CURRENCY*")

    Returns:
        str
    """
    if replace_with is None:
        for k, v in CURRENCIES.items():
            text = text.replace(k, " "+v+" ")
        return text
    else:
        return CURRENCY_REGEX.sub(replace_with, text)


def convert_currency_numbers(text):
    r=re.compile('[{0}]+'.format(''.join(CURRENCIES_LONG.keys())))
#    r=re.compile('[{0}]+'.format(''.join(CURRENCIES_LONG.keys())) + '([0-9][0-9.]*)([kKmMbBgG]?)')
    resultobj = r.search(text)
    if resultobj:
       for k, v in CURRENCIES_LONG.items():
           text = text.replace(k, ' '+v+ ' ')
    return text

def convert_currency_name(text):
    r=re.compile('[{0}]+'.format(''.join(CURRENCIES_abbr.keys())) + '([0-9][0-9.]*)([kKmMbBgG]?)')
    resultobj = r.search(text)
    if resultobj:
       for k, v in CURRENCIES_abbr.items():
#           print text
#           print k, v
           text = text.replace(k, ' '+ v+ ' ')
    return text


def get_unit(ustr):
    if ustr == '': return 'u'
    return ustr.lower()


def convert_kmbg_zero(text):
    r=re.compile(r'\b([1-9][0-9.]*)([kKmMbBgG]?)\b')
    units={'k':1000,'m':1000000,'g':1000000000,'b':1000000000,'u':1}
    result=r.search(text)
    if result:
       if result.group() != '401k' and result.group(2) != ''and result.group().count(".") <2 :
          long_number = str(int(float(result.group(1))*units[get_unit(result.group(2))]))
          text = r.sub(long_number, text)
    return text 


def remove_comma_number(text):
    return re.sub(r'(?:(\d+?),)',r'\1', text)
    '''
    r=re.compile(r'\b[1-9][0-9,.]{0,}')
    searchobj = r.search(text)
    print searchobj.groups()
    if searchobj:
       print 'test'
       numbers_no_comma = re.sub("[^\d\.]", "", searchobj.group())
       text = r.sub(numbers_no_comma, text)
    return text
    '''


def remove_punct(text, marks=None):
    """
    Remove punctuation from ``text`` by replacing all instances of ``marks``
    with an empty string.

    Args:
        text (str): raw text
        marks (str): If specified, remove only the characters in this string,
            e.g. ``marks=',;:'`` removes commas, semi-colons, and colons.
            Otherwise, all punctuation marks are removed.

    Returns:
        str

    .. note:: When ``marks=None``, Python's built-in :meth:`str.translate()` is
        used to remove punctuation; otherwise,, a regular expression is used
        instead. The former's performance is about 5-10x faster.
    """
    if marks:
        return re.sub('[{}]+'.format(re.escape(marks)), ' ', text, flags=re.UNICODE)
    else:
        if isinstance(text, unicode_):
            return text.translate(PUNCT_TRANSLATE_UNICODE)
        else:
            return text.translate(None, PUNCT_TRANSLATE_BYTES)


def am_pm_convert(text):

    token_pattern=r"\b([0-9]+)\s*(a.m.|p.m.)"
    r = re.compile(token_pattern)
    obj = r.search(text)
    if obj:
       text = re.sub(token_pattern, r"\1"+re.sub(r'\.', "", obj.group(2)),text)
    return text


## synonym replacer
replacer = CsvWordReplacer('%s/synonyms.csv' % config.data_folder)


def clean_text(l, drop_html_flag=False):
    if l !="":
#       print line
#       print type(line)
       if drop_html_flag:
          l = drop_html(l)
       l = l.lower()
       l = fix_bad_unicode(l, normalization='NFC')
#    l = ftfy.fix_text(l,normalization='NFC')
#    l = transliterate_unicode(l)

   
       l = remove_accents(l, method='unicode')
       l = replace_currency_symbols(l)
       l = convert_currency_name(l)
    
     #l = l.replace('-', ' ')
    # remove comma in numbers
       l = remove_comma_number(l)
    # convert currency symbols to words
    #l = convert_currency_numbers(l)
    # convert kmgb to 0
       l = convert_kmbg_zero(l)
    # unpack contractions
       l = unpack_contractions(l)
       l = remove_punct(l,marks='()[];,:?\'\"!-')
       l = l.split(" ")
        ## replace synonyms
       l = replacer.replace(l)
       l = " ".join(l)
       l = am_pm_convert(l)
       l = l.replace('/', ' or ')
       l = remove_punct(l,marks='.')
       l = normalize_whitespace(l)

    return l    
###################
## Drop html tag ##
###################
def drop_html(html):
    return BeautifulSoup(html,"lxml").get_text(separator=" ",strip=True)
