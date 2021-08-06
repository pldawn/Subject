import torch
from torch.nn.functional import normalize
from torch.nn import Module


class TransD(Module):
    def __init__(self, embedding_dim, entity_nums, meta_entity_nums, relation_nums, meta_relation_nums):
        super(TransD, self).__init__()
        # 记录参数
        self.embedding_dim = embedding_dim
        self.entity_nums = entity_nums
        self.meta_entity_nums = meta_relation_nums
        self.relation_nums = relation_nums
        self.meta_relation_nums = meta_relation_nums

        # 定义embeddings
        self.entity_embedding_layer = torch.nn.Embedding(entity_nums, embedding_dim)
        self.relation_embedding_layer = torch.nn.Embedding(relation_nums, embedding_dim)
        self.meta_entity_embedding_layer = torch.nn.Embedding(meta_entity_nums, embedding_dim)
        self.meta_relation_embedding_layer = torch.nn.Embedding(meta_relation_nums, embedding_dim)

        # 初始化
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.children():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, head_ids, meta_head_ids, relation_ids, meta_relation_ids, tail_ids, meta_tail_ids, **kwargs):
        # 得到embeddings
        head_embeddings = self.entity_embedding_layer(head_ids)
        head_meta_embeddings = self.meta_entity_embedding_layer(meta_head_ids)
        relation_embeddings = self.relation_embedding_layer(relation_ids)
        relation_meta_embeddings = self.meta_relation_embedding_layer(meta_relation_ids)
        tail_embeddings = self.entity_embedding_layer(tail_ids)
        tail_meta_embeddings = self.meta_entity_embedding_layer(meta_tail_ids)

        # 计算转移矩阵
        batch_size = head_embeddings.shape[0]
        embedding_dim = head_embeddings.shape[1]

        proj_relation_meta_embeddings = relation_meta_embeddings.view(batch_size, embedding_dim, 1)
        proj_head_meta_embeddings = head_meta_embeddings.view(batch_size, 1, embedding_dim)
        proj_tail_meta_embeddings = tail_meta_embeddings.view(batch_size, 1, embedding_dim)
        proj_eye = torch.eye(embedding_dim, embedding_dim, dtype=head_embeddings.dtype, device=self.device)

        head_transformation = torch.matmul(proj_relation_meta_embeddings, proj_head_meta_embeddings) + proj_eye
        tail_transformation = torch.matmul(proj_relation_meta_embeddings, proj_tail_meta_embeddings) + proj_eye

        # 计算score
        proj_head_embeddings = normalize(torch.matmul(head_transformation, head_embeddings))
        proj_tail_embeddings = normalize(torch.matmul(tail_transformation, tail_embeddings))

        difference = proj_head_embeddings + relation_embeddings - proj_tail_embeddings
        score = torch.norm(difference, 1)

        return score

    def get_entity_embeddings(self, ids):
        return self.entity_embedding_layer(ids)

    def get_relation_embeddings(self, ids):
        return self.relation_embedding_layer(ids)

    def get_meta_entity_embeddings(self, ids):
        return self.meta_entity_embedding_layer(ids)

    def get_meta_relation_embeddings(self, ids):
        return self.meta_relation_embedding_layer(ids)
