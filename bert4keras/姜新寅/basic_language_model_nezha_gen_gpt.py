#! -*- coding: utf-8 -*-
# 基本测试：中文GPT模型，base版本，华为开源的
# 权重链接: https://pan.baidu.com/s/1-FB0yl1uxYDCGIRvU1XNzQ 提取码: xynn
# 参考项目：https://github.com/bojone/chinese-gen

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
# from bert4keras.snippets import uniout

config_path = 'S:/python/chinese_nezha_gpt_L-12_H-768_A-12/config.json'
checkpoint_path = 'S:/python/chinese_nezha_gpt_L-12_H-768_A-12/gpt.ckpt'
dict_path = 'S:/python/chinese_nezha_gpt_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    segment_vocab_size=0,  # 去掉segmeng_ids输入
    application='lm',
)  # 建立模型，加载权重


class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return self.last_token(model).predict(token_ids)

    def generate(self, text, n=5, output_topk=5, topp=0.95):
        token_ids = tokenizer.encode(text)[0][:-1]
        # results = self.random_sample([token_ids], n, topk=topk, topp=topp)  # 基于随机采样
        results = self.beam_search([token_ids], 9, output_topk=output_topk)
        for ids in results:
            print(text + tokenizer.decode(ids))
        return [text + tokenizer.decode(ids) for ids in results]


article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,  # 511是中文句号
    maxlen=256,
    minlen=128
)

print(article_completion.generate(u'抄袭检测'))