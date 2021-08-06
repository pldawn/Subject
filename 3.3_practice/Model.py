from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model


class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

    def forward(self, input_ids=None, past_key_values=None, token_type_ids=None, labels=None, title_id=None):
        """
        前向函数，计算GPT2预测结果值
        Args:
            input_ids: 输入序列在词表中的索引序列，size:[batch_size, sequence_length]
            past_key_values: 包含由模型预先计算好的隐藏状态，一般使用在预测阶段，用于加速顺序解码，防止重复计算前面计算过的token
            token_type_ids: 用于区分输入序列中content和title的分隔符序列，size:[batch_size, sequence_length]
            labels: 标签序列，size:[batch_size, sequence_length]，一般情况下，与input_ids相同
            title_id: title部分分隔符的id
        Returns:

        """
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]

        if labels is not None:
            if title_id is None or token_type_ids is None:
                raise Exception("当labels不为None时， title_id和token_type_ids均不可以为None")

            mask = (token_type_ids == title_id).long()
            labels = labels * mask
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num

            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
