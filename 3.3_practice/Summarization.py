import copy
import torch
from transformers import AutoTokenizer
from Configure import Configure
from Model import GPT2LMHeadModel


class Summarizer:
    def __init__(self, model_path, tokenizer_path, configure=None):
        self.configure = Configure(configure)

        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if not torch.cuda.is_available():
            self.configure["device"] = "cpu"

        self.device = torch.device(self.configure["device"])
        self.model.to(self.device)
        self.model.eval()

    def generate_summary(self, one_sentence: str, greedy=False):
        """
        Args:
            one_sentence: str
            greedy: bool
        """
        if len(one_sentence) == 0 or one_sentence is None:
            return []

        # 对新闻正文进行预处理，并判断如果超长则进行截断
        content_tokens = self.tokenizer.tokenize(one_sentence)
        content_tokens = content_tokens[:self.configure["max_length"] - self.configure["max_summary_length"] - 3]

        # 获取content_id、title_id、unk_id、sep_id值
        content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        title_id = self.tokenizer.convert_tokens_to_ids("[Title]")
        unk_id = self.tokenizer.convert_tokens_to_ids("[UNK]")
        sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")

        # 将tokens索引化，变成模型所需格式
        content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(content_tokens)

        # 贪婪解码策略下，只生成概率最大的title
        if greedy:
            summary_nums = 1
        else:
            summary_nums = self.configure["summary_nums"]

        # 将input_ids和token_type_ids进行扩充，扩充到需要预测标题的个数，即batch_size
        input_ids = [copy.deepcopy(input_ids) for _ in range(summary_nums)]
        token_type_ids = [[content_id] * len(content_tokens) for _ in range(summary_nums)]

        # 将input_ids和token_type_ids变成tensor
        input_tensors = torch.tensor(input_ids).long().to(self.device)
        token_type_tensors = torch.tensor(token_type_ids).long().to(self.device)
        next_token_type = torch.tensor([[title_id] for _ in range(summary_nums)]).long().to(self.device)

        generated_summaries = []  # 用于存放每一步解码的结果
        finish_set = set()  # 用于存放，完成解码序列的序号

        with torch.no_grad():
            # 生成摘要，最长不超过max_summary_length
            for _ in range(self.configure["max_summary_length"]):
                outputs = self.model(input_ids=input_tensors, token_type_ids=token_type_tensors)
                next_token_logits = outputs[0][:, -1, :]

                # 对batch_size进行遍历，将词表中出现在序列中的词的概率进行惩罚
                repetition_check_start_ind = max(0, len(generated_summaries) - self.configure["repetition_window"])

                for index in range(summary_nums):
                    for token_id in set([token_ids[index] for token_ids in generated_summaries[repetition_check_start_ind:]]):
                        next_token_logits[index][token_id] /= self.configure["repetition_penalty"]

                # 对batch_size进行遍历，将词表中的UNK的值设为无穷小
                for next_token_logit in next_token_logits:
                    next_token_logit[unk_id] = -float("Inf")

                if greedy:
                    # 贪婪策略下，仅取概率最大的token
                    next_tokens = torch.argmax(torch.softmax(next_token_logits, dim=-1), dim=-1, keepdim=True)
                else:
                    # 非贪婪策略下，使用top_k_top_p策略对logits进行过滤，在过滤后的logits上依概率取得token
                    filter_logits = self.top_k_top_p_filtering(next_token_logits)
                    next_tokens = torch.multinomial(torch.softmax(filter_logits, dim=-1), num_samples=1)

                # 判断如果哪个序列的预测标记为sep_id时，则加入到finish_set
                for index, token_id in enumerate(next_tokens[:, 0]):
                    if token_id == sep_id:
                        finish_set.add(index)

                # 判断，如果finish_set包含全部的序列序号，则停止预测；否则继续预测
                finish_flag = True

                for index in range(summary_nums):
                    if index not in finish_set:
                        finish_flag = False
                        break

                if finish_flag:
                    break

                # 将预测标记添加到generated中
                generated_summaries.append([token.item() for token in next_tokens[:, 0]])

                # 将预测结果拼接到input_tensors和token_type_tensors上，继续下一次预测
                input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
                token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)

        # 用于存储预测结果
        candidate_responses = []

        # 对摘要数量进行遍历，并将token_id变成对应汉字
        for index in range(summary_nums):
            responses = []

            # 对摘要长度进行遍历
            for token_index in range(len(generated_summaries)):
                # 判断，当出现sep_id时，停止在该序列中添加token
                if generated_summaries[token_index][index] != sep_id:
                    responses.append(generated_summaries[token_index][index])
                else:
                    break

            # 将token_id序列变成汉字序列，去除"##"，并将[Space]替换成空格
            decoded = self.tokenizer.convert_ids_to_tokens(responses)
            candidate_responses.append("".join(decoded).replace("##", "").replace("[space]", " "))

        return candidate_responses

    def top_k_top_p_filtering(self, logits):
        """
        top_k或top_p解码策略，仅保留top_k个或累积概率到达top_p的标记，其他标记设为filter_value，后续在选取标记的过程中会取不到值设为无穷小。

        """
        assert logits.dim() == 2

        # 获取top_k和字典大小中较小的一个，也就是说，如果top_k大于字典大小，则取字典大小个标记
        top_k = min(self.configure["top_k"], logits[0].size(-1))

        # 如果top_k不为0，则将在logits中保留top_k个标记
        if top_k > 0:
            for logit in logits:
                indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
                logit[indices_to_remove] = -float("Inf")

        # 如果top_p不为0，则将在logits中保留概率值累积达到top_p的标记
        top_p = self.configure["top_p"]

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # 对排序后的结果使用softmax归一化，再获取累积概率序列
            # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = torch.as_tensor(cumulative_probs > top_p, dtype=torch.int32)

            # 将索引向右移动，使第一个标记总是被保留
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for index, logit in enumerate(logits):
                indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
                logit[indices_to_remove] = -float("Inf")

        return logits


if __name__ == '__main__':
    summary_model_path = ""
    summary_tokenizer_path = ""

    agent = Summarizer(summary_model_path, summary_tokenizer_path)
    content = "澳门2014年现金分享计划将于7月2日正式实施。届时，澳门特区永久性居民及非永久性居民将分别获发9000和5400澳门元。澳门特区政府此项财政开支约为56.59亿。为市民共享经济发展成果，澳门08年起推出现金分享计划。"  # input("输入的新闻正文为:")
    print(f"原文: {content}")

    summaries = agent.generate_summary(content, greedy=False)
    for i, title in enumerate(summaries):
        print("生成的第{}个标题为：{}".format(i + 1, title))
