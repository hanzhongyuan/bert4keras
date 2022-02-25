#! -*- coding: utf-8 -*-
# 基本测试：中文GPT2_ML模型
# 介绍链接：https://kexue.fm/archives/7292

import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder
# from bert4keras.snippets import uniout

config_path = 'E:/bert_model/model/GPT2_ML/config.json'
checkpoint_path = 'E:/bert_model/model/GPT2_ML/model.ckpt-100000'
dict_path = 'E:/bert_model/model/GPT2_ML/vocab.txt'

tokenizer = Tokenizer(
    dict_path, token_start=None, token_end=None, do_lower_case=True
)  # 建立分词器

model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, model='gpt2_ml'
)  # 建立模型，加载权重


class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = np.concatenate([inputs[0], output_ids], 1)
        return self.last_token(model).predict(token_ids)

    def beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1):
        """beam search解码
        说明：这里的topk即beam size；
        返回：最优解码序列。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids, output_scores = self.first_output_ids, np.zeros(1)      # np.zeros(1)给定维度的全零数组
        results = []
        key = 0
        for step in range(self.maxlen):
            # print(step)
            scores, states = self.predict(
                inputs, output_ids, states, temperature, 'logits'
            )  # 计算当前得分
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分      reshape(a, b)以a行b列的数组形式显示
            indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
            indices_1 = indices // scores.shape[1]  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = np.concatenate([output_ids[indices_1], indices_2],
                                        1)  # 更新输出
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  # 更新得分
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best = output_scores.argsort()[-5:][::-1]  # 得分最大的5个
                for b in best:
                    if is_end[b] and end_counts[b] >= min_ends:  # 如果已经终止
                        results.append(output_ids[b])
                        key += 1
                        if key == 5:
                            return results
                    # else:  # 否则，只保留未完成部分
                    #     flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                    #     if not flag.all():  # 如果有已完成的
                    #         inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                    #         output_ids = output_ids[flag]  # 扔掉已完成序列
                    #         output_scores = output_scores[flag]  # 扔掉已完成序列
                    #         end_counts = end_counts[flag]  # 扔掉已完成end计数
                    #         topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        best = output_scores.argsort()[-5:][::-1]
        for ib in best:
            results.append(output_ids[ib])
            key += 1
            if key == 5:
                return results
        return results

    def generate(self, text, topk=6):
        token_ids, _ = tokenizer.encode(text)
        results = self.beam_search([token_ids], topk=topk)  # 基于随机采样
        # return [text + tokenizer.decode(ids) for ids in results]
        return [text + tokenizer.decode(ids) for ids in results]


article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,  # 511是中文句号
    maxlen=256,
    minlen=128
)

print(article_completion.generate(u'抄袭检测'))

