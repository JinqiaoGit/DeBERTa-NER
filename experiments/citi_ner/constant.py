# _*_ coding: utf-8 _*_

# the label of padding mark
PADDING_LABEL = '<PAD>'

# labels are based BIO marking method
LABEL2IX = {
    "O": 0,
    "B-TICKER": 1,
    "I-TICKER": 2,
    "B-COMPANY": 3,
    "I-COMPANY": 4,
    "B-COUNTRY": 5,
    "I-COUNTRY": 6
}

# the map from idx to string label
IX2LABEL = {v: k for k, v in LABEL2IX.items()}
