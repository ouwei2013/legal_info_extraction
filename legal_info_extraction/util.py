
import os
import regex as re

import numpy as np
import spacy
from bs4 import BeautifulSoup
from spacy import displacy
from spacy.lang.zh import Chinese
from spacy.language import Language
from spacy.tokens import Span

spacy_zh = Chinese()


def overlap(ent1, ent2):
    """判断两个entity是不是有重叠

    Args:
        ent1 (spacy entity): 
        ent2 (spacy entity ): 

    Returns:
        bool
    """
    tmp = set(range(ent1.start, ent1.end)).intersection(
        set(range(ent2.start, ent2.end)))
    if len(tmp) > 0:
        return True
    else:
        return False


def expand(d, ent):
    """对一个entity的结束位置扩展到最近的一个句号。

    Args:
        d (str): 
        ent (list )



    """
    st, ed, label, txt = ent
    for i in range(ed, len(d)):
        if d[i] != '。':
            i = i+1
        else:
            break
    return st, i, label, d[st:i]


def filter(rows):
    """过滤掉结果中内容一样的entity

    Args:
        rows : [[ent1_start,ent1_end,ent1_type,ent1_txt],[ent2_start,ent2_end,ent2_type,ent2_txt]...]

    Returns:
        rows: 同输入，但是去掉了内容一样的entity
    """
    filtered = []
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j:
                if j < len(rows)-1:
                    continue
                else:
                    filtered.append(rows[j])
            else:
                if rows[i][-1] in rows[j][-1] and rows[i][2] == rows[j][2]:
                    break
                if j == len(rows)-1:
                    filtered.append(rows[i])
    return filtered


def create_spacy_doc_from_ents(doc, doc_ents):
    doc = spacy_zh(doc)  # 把原始文本包装成spacy的doc类，方便处理
    all_ent_st_idx = []
    ents = []
    for ent in doc_ents:
        if ent[0] not in all_ent_st_idx:
            all_ent_st_idx.append(ent[0])
            ents.append(Span(doc, ent[0], ent[1], ent[2]))
    doc.ents = ents
    doc[0].is_sent_start = True
    doc[-1].is_sent_start = False
    for idx, t in enumerate(doc):
        if t.text == '。' and idx < len(doc)-1:
            doc[idx+1].is_sent_start = True
    return doc
