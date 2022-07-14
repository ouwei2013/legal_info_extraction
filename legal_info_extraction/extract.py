
import os
import regex as re

import numpy as np
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from spacy import displacy
from spacy.lang.zh import Chinese
from spacy.language import Language
from spacy.tokens import Span
from legal_info_extraction.util import *


class LegalInfoExtractor:

    def __init__(self, ner_model_path) -> None:
        self.ner = spacy.load(ner_model_path)
        self.doc_wrapper = Chinese()

        self.rqst_patn1 = re.compile(
            '(请求|要求|申请).{,5}(撤销|判令|确认|(执行[^人])|责成|判决|责令)')
        self.rqst_patn2 = re.compile(
            '提出(撤销|判令|责令|确认|执行|责成|判决|撤回).{,20}(请求|要求|申请|起诉)')

    def _combine(self, docs):
        """transformer先对句子进行NER识别。本函数把处理后的句子组装回原来的长文本

        Args:
            docs : 已经处理过的句子

        Returns:
            orig_doc: str类型，也就是组装后的原始文本
            orig_ents: List类型，对应orig_doc里面的NE
        """
        orig_doc = ''
        orig_ents = []
        offset = 0
        for idx, d in enumerate(docs):
            offset = len(orig_doc)
            orig_doc = orig_doc+d.text

            # 根据正则表达式，判断是否是"诉求"
            m = self.rqst_patn1.search(d.text)
            if not m:
                m = self.rqst_patn2.search(d.text)
            if m:
                st, _ = m.span()
                d.ents = [d.char_span(st, len(d.text)-1, '诉求')]

            for ent in d.ents:
                ent_txt = ent.text
                # 如果本句话里没有任何汉字，忽略这句话
                if not re.search('[\u4e00-\u9fa5]{2,}', ent_txt):
                    continue
                label = ent.label_
                if label == '行政主体':  # 分析NER模型结果，发现它经常把行政主体和理由搞混
                    if len(ent_txt) > 30:  # 所以当一个识别出来的”行政主体“太长，实际上它应该是”理由“
                        label = '理由'
                if label == '诉求':  # 分析NER模型结果，发现它经常把诉求和行政处罚决定搞混
                    # 所以对识别出来的“诉求“，也进行正则表达式匹配，看看是不是真的是诉求
                    if not self.rqst_patn1.search(d.text):
                        label = '行政处罚决定'
                if '理由' in label:  # 分析NER模型结果，发现它经常把带有”是否“ 和'发生法律效力'的句子识别称理由，实际上它们不是理由
                    if '是否' in ent_txt or idx == len(docs)-1 or '发生法律效力' in ent_txt:
                        continue

                st = ent.start + offset
                ed = ent.end+offset
                orig_ents.append([st, ed, label, ent_txt])
        return orig_doc, orig_ents

    def _seg_doc(self, doc):
        """
        文本一般有这样的结构：原告认为xxxxxxxxxx 。被告认为xxxxxxx。一审法院认为xxxxxxx。本院认为xxxxxxx。
        把判决文书根据原告、被告、法院的意见进行分割


        Args:
            doc (spacy doc ): 判决文本

        Returns:
            section_idx: 法院意见的开始和结束的index
        """
        claim_idx = [0]  # 用来存放各方主张的开始和结束的index
        claim_subj = ['']  # 用来存放各方主张的主语
        court_patn = re.compile(
            '((一审|原审|二审|三审|本院)|[\u4e00-\u9fa5]{,10}法院)[\u4e00-\u9fa5]{,4}认为')  # 匹配法院的意见
        for m in re.finditer('''[\u4e00-\u9fa5，、；""]{2,20}(辩称|诉称|再审称|认为)[：，]''', doc):  # 匹配各方的意见
            st, _ = m.span()
            claim_idx.append(st)
            claim_subj.append(m.group())
        if len(claim_idx) == 1:  # 如果没有找到任何意见段落，那么不进行分割
            claim_idx.append(len(doc)-1)
            claim_subj.append('')
        section_idx = dict()
        for idx, (i, subj) in enumerate(zip(claim_idx, claim_subj)):
            if idx == 0:  # 在各方主张开放之前，一般都是介绍案情的基本信息，比如案由
                section_idx['基本信息'] = list(range(0, claim_idx[idx+1]))
            else:
                # 只保留法院意见的开始和结束位置
                m = court_patn.search(subj)  # 判断是不是法院的意见
                if m:
                    subj = m.group()
                    if idx < len(claim_idx)-1:
                        section_idx[subj] = list(range(i, claim_idx[idx+1]))
                    else:
                        section_idx[subj] = list(range(i, len(doc)))
        return section_idx

    def _find_applicants(self, doc):
        """从判决文书的开头中根据规则找到原告

        Args:
            doc (spacy.doc): 判决文书

        Returns:
            applications: 原告的开始、结束位置、以及原告的名称
        """
        applicants = []
        for sen in doc.sents:
            txt = sen.text
            m = re.search(
                '((?!<被)上诉人|原告|(?!<被)申请人|(?!<被)申请执行人|(?!<被)起诉人)([(（][\u4e00-\u9fa5，、:：]{2,20}[)）])?[:：\s]?', txt)
            if m:
                if len(sen.ents) == 0:
                    st, ed = re.search(
                        '(?<=%s[^\u4e00-\u9fa5]{,3})[\u4e00-\u9fa5]{1,20}' % re.escape(m.group()), txt).span()
                    applicants.append(
                        [st+sen.start, ed+sen.end, '原告', txt[st:ed]])
                    m = re.search(
                        '(?<=简称[^\u4e00-\u9fa5]{,3})[\u4e00-\u9fa5]{2,20}', txt)
                    if m:
                        st, ed = m.span()
                        applicants.append(
                            [st+sen.start, ed+sen.end, '原告简称', txt[st:ed]])
                else:
                    applicants = applicants+[[ent.start, ent.end, ent.label_, ent.text]
                                             for ent in sen.ents if ent.label_ == '行政主体']
                return applicants
        return applicants

    def _find_defenders(self, doc):
        """从判决文书的开头中根据规则找到被告

        Args:
            doc (spacy.doc): 判决文书

        Returns:
            defenders: 被告的开始、结束位置、以及原告的名称
        """
        defenders = []
        for sen in doc.sents:
            txt = sen.text
            m = re.search(
                '(被上诉人|被执行人|被申请人|被申请执行人|被告|被诉人|被起诉人)([(（][\u4e00-\u9fa5，、:：]{2,20}[)）])?[:：\s]?', txt)
            if m:
                if len(sen.ents) == 0:
                    st, ed = re.search(
                        '(?<=%s[^\u4e00-\u9fa5]{,3})[\u4e00-\u9fa5]{1,20}' % re.escape(m.group()), txt).span()
                    defenders.append(
                        [st+sen.start, ed+sen.end, '被告', txt[st:ed]])
                    m = re.search(
                        '(?<=简称[^\u4e00-\u9fa5]{,3})[\u4e00-\u9fa5]{2,20}', txt)
                    if m:
                        st, ed = m.span()
                        # tmp=list(d.ents)
                        defenders.append(
                            [st+sen.start, ed+sen.end, '被告简称', txt[st:ed]])
                else:
                    defenders = defenders + \
                        [[ent.start, ent.end, ent.label_, ent.text]
                            for ent in sen.ents if ent.label_ == '行政主体']

                return defenders

        return defenders

    def _find_cause(self, doc, applicants, defenders):
        """从文本中根据规则和模型输出寻找案由

        Args:
            doc (spacy.doc): 判决文书
            applicants (原告)): 
            defenders (被告)): 

        Returns:
            _type_: _description_
        """
        psn_orgs = ['被上诉人', '被执行人',
                    '被申请人', '被申请执行人',
                    '被告', '被诉人', '上诉人',
                    '原告', '申请人', '申请执行人']
        cause_ents = [ent for ent in doc.ents if ent.label_ == '案由']
        non_cause_ents = [ent for ent in doc.ents if ent.label_ != '案由']
        psn_orgs = psn_orgs + [ent[-1] for ent in applicants +
                               defenders if ent[-2] in ['行政主体', '原告', '被告', '原告简称', '被告简称']]
        for sen in doc.sents:
            if '一案' in sen.text:  # 如果一句话里有"一案" , 那么这句话里一定包含了案由

                context = re.search('[^。，]*一案', sen.text).group()
                # 先检查"一案"前面，有没有模型识别出来的NE
                # 如果有，那么这个NE一定是案由
                for ent in sen.ents:
                    if ent.label_ in ['行政主体', '原告', '被告', '原告简称', '被告简称']:
                        continue
                    if re.search('%s.{,3}一案' % re.escape(ent.text), context):
                        ent.label_ = '案由'
                        non_cause_ents = [
                            tmp for tmp in non_cause_ents if tmp.start != ent.start]
                        doc.ents = non_cause_ents+[ent]
                        return doc
                # 如果模型没有在这句话里检查到任何案由，那么进行规则查找
                # 查找方法就是找出原告和被告名称与"一案“之间的文字，作为案由
                psn_orgs = '|'.join(nm for nm in psn_orgs)
                psn_orgs = '(%s)' % psn_orgs
                case_txt = re.sub(psn_orgs, '_', context)
                case_txt = case_txt.split('_')[-1]
                m = re.search('[\u4e00-\u9fa5《》]+(?=一案)', case_txt)
                if m:
                    case_txt = m.group()
                else:
                    case_txt = context
                st = sen.start+sen.text.index(case_txt)
                ed = st + len(case_txt)
                cause_ents = [doc.char_span(st, ed, label='案由')]
                non_cause_ents = [
                    tmp for tmp in non_cause_ents if not overlap(tmp, cause_ents[0])]

                doc.ents = cause_ents+non_cause_ents
                return doc
        return doc

    def _organize_ents(self, doc, section_idx):
        """对NE按照原告、被告、基本信息、诉求、法院意见等进行整理

        Args:
            doc (spacy.doc): 判决文书
            section_idx : 基本信息、各法院意见的开始、结束index

        Returns:
            vdcts: dict类型，key为原告、被告、基本信息、诉求、法院意见等,val为[[NE1_start,NE_end,NE1_type,NE1_text],...]
        """
        vdcts = dict()
        applicants = self._find_applicants(doc)
        defenders = self._find_defenders(doc)
        if applicants:
            vdcts['原告'] = applicants
        if defenders:
            vdcts['被告'] = defenders

        doc = self._find_cause(doc, applicants, defenders)

        for ent in doc.ents:
            if ent.label_ == '诉求':
                tmp = vdcts.get('诉求', [])
                tmp.append([ent.start, ent.end, ent.label_, ent.text])
                vdcts['诉求'] = tmp
            else:
                for k, v in section_idx.items():
                    if ent.start in v:
                        tmp = vdcts.get(k, [])
                        tmp.append([ent.start, ent.end, ent.label_, ent.text])
                        vdcts[k] = tmp
        return vdcts

    def _enhance_result(self, doc, vdcts):
        """对结果进行进一步的优化

        Args:
            doc (spacy.doc): 判决文书
            vdcts :  dict类型，key为原告、被告、基本信息、诉求、法院意见等,val为[[NE1_start,NE_end,NE1_type,NE1_text],...]

        Returns:
            vdcts: 优化后的vdcts
        """
        for k, v in vdcts.items():
            rows = []
            for i in v:
                st, ed, label, txt = i
                if k == '基本信息':
                    if label == '理由':  # 基本信息区域内的“理由”，一般都是行为
                        label = '行为'
                else:
                    if label in ['行为', '理由']:  # 法院意见区域内的“行为”，一般都是理由
                        label = '理由'
                    # ner模型识别的理由可能只截取了一句话中间的部分信息，我们发现对把理由从句中扩展到句尾，不容易丢失信息
                    st, ed, label, txt = expand(doc, [st, ed, label, txt])
                rows.append([st, ed, label, txt])
            rows = filter(rows)
            vdcts[k] = rows
        return vdcts

    def extract(self, doc):
        """从判决文书中提取信息

        Args:
            doc (str): 判决文书

        Returns:
            result:  dict类型，key为原告、被告、基本信息、诉求、法院意见等,val为[[NE1_start,NE_end,NE1_type,NE1_text],...]
        """
        soup = BeautifulSoup(doc)
        txt = soup.get_text()
        # 判决文书的开头是该文书的id，例如“ xx法院xx号”，去掉编号的法院信息以避免对模型的影响
        txt = re.sub('^.{,20}法院', '', txt)
        txt = re.sub('\s', '', txt)
        sentences = txt.split('。')
        # 把文书按照句号分割
        sentences = [s.strip()+'。' for s in sentences if len(s.strip()) > 0]
        docs = list(self.ner.pipe(sentences))  # 把分割后的句子输入到ner模型，进行ne识别
        orig_doc, orig_ents = self._combine(docs)  # 把分割的句子以及识别结果组装成原来的判决文本
        orig_doc = create_spacy_doc_from_ents(
            orig_doc, orig_ents)  # 把原始文本包装成spacy的doc类，方便处理
        section_idx = self._seg_doc(orig_doc.text)
        vdcts = self._organize_ents(orig_doc, section_idx)
        results = self._enhance_result(orig_doc, vdcts)
        return results
