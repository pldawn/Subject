import math
import torch
from Configure import Configure
from Transformer import TransformerEncoderForSequenceClassification
from Transformer import TransformerEncoderForTokenClassification
from Dataset import DataIterator
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig


class Distiller:
    def __init__(self, configure=None):
        self.configure = Configure(configure)

    def distill(self,
                task_tag,
                corpus_tag_or_path,
                teacher_model_or_path,
                tokenizer_path,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                is_split_into_words=None,
                labels_map=None,
                max_length=512,
                preprocess_fn=None,
                **kwargs):
        """
        :param task_tag: str, 有效值是L3TC, L3SC, L6TC, L6SC
        :param corpus_tag_or_path: str, 训练数据集路径，应满足标准数据集的格式
        :param teacher_model_or_path: Optional[str, RobertaModel], 模型路径或模型实例
        :param tokenizer_path: str, 分词器的路径
        :param batch_size: int, default=1
        :param shuffle: bool, default=False
        :param num_workers: int, default=0, 数据集读取进程数，数据集较大时，可以设置等于cpu数量
        :param is_split_into_words: Optional[int], default=None
        :param kwargs: nlptoolkit.Component.Model.TransformerForXXXClassification类的实例化参数
        :param labels_map: dict
        :param max_length: int, default=512
        :param preprocess_fn: FunctionType, 预处理或加噪
        :return:
        """
        # 验证task_tag的有效性
        assert task_tag in ("L3SC", "L3TC", "L6SC", "L6TC"), "model_tag must be one of L3SC/L3TC/L6SC/L6TC."

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 重新设定teacher model的输出结果，要求模式输出bert结构中的所有hidden states和attentions
        if type(teacher_model_or_path) == str:
            teacher_model = torch.load(teacher_model_or_path)
        else:
            teacher_model = teacher_model_or_path

        teacher_model.eval()
        teacher_model.config.output_attentions = True
        teacher_model.config.output_hidden_states = True

        # 构造student model
        num_labels = teacher_model.num_labels
        student_model = self.generate_student_model(task_tag, num_labels, **kwargs)
        student_model.cuda(device)

        # 构造优化器
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

        # 构造数据集
        data_iterator = DataIterator.from_corpus(
            corpus_tag_or_dir=corpus_tag_or_path,
            tokenizer_path=tokenizer_path,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            is_split_into_words=is_split_into_words,
            labels_map=labels_map,
            max_length=max_length,
            preprocess_fn=preprocess_fn,
        )

        if not hasattr(data_iterator["train_iter"], "__len__"):
            num_steps = math.ceil(data_iterator["config"]["train_size"] * self.configure["distiller_train_config"]["num_epochs"] / batch_size)
            self.configure["distiller_train_config"]["num_steps"] = num_steps

            ckpt_steps = math.ceil(data_iterator["config"]["train_size"] / batch_size)
            self.configure["training_config"]["ckpt_steps"] = ckpt_steps

        # 根据任务标签设定matches
        intermediate_matches = self.generate_task_matches(task_tag)
        self.configure["distillation_config"]["intermediate_matches"] = intermediate_matches

        # 设定adaptor，teacher和student采用相同的adaptor
        adaptor = self.generate_task_adaptor()

        # 初始化蒸馏器
        distiller = GeneralDistiller(
            train_config=TrainingConfig(**self.configure["training_config"], device=device),
            distill_config=DistillationConfig(**self.configure["distillation_config"]),
            model_T=teacher_model,
            model_S=student_model,
            adaptor_T=adaptor,
            adaptor_S=adaptor,
        )

        # 进行蒸馏，蒸馏模型和蒸馏过程日志的保存路径在Configure中
        distiller.train(
            optimizer=optimizer,
            dataloader=data_iterator["train_iter"],
            **self.configure["distiller_train_config"]
        )

    def generate_student_model(self, task_tag, num_labels, **kwargs):
        if task_tag.startswith("L3"):
            layer_num = 3
        elif task_tag.startswith("L6"):
            layer_num = 6
        else:
            raise ValueError("task_tag must start with L3 or L6")

        kwargs.update({
            "output_attentions": True,
            "output_hidden_states": True,
        })

        if task_tag.endswith("TC"):
            model = TransformerEncoderForTokenClassification(num_labels=num_labels, L=layer_num, **kwargs)

        elif task_tag.endswith("SC"):
            model = TransformerEncoderForSequenceClassification(num_labels=num_labels, L=layer_num, **kwargs)
        else:
            raise ValueError("task_tag must end with SC or TC")

        return model

    def generate_task_matches(self, task_tag):
        if task_tag.startswith("L3"):
            matches = self.configure["matches"]["L3_hidden_smmd"] + self.configure["matches"]["L3_hidden_mse"]

        elif task_tag.startswith("L6"):
            matches = self.configure["matches"]["L6_hidden_smmd"] + self.configure["matches"]["L6_hidden_mse"]

        else:
            raise ValueError("task_tag must start with L3 or L6")

        return matches

    def generate_task_adaptor(self):
        def adaptor(batch, model_outputs):
            result = {
                "inputs_mask": batch["attention_mask"],
                "logits_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "losses": model_outputs["loss"],
                "logits": model_outputs["logits"],
                "hidden": model_outputs["hidden_states"],
                "attention": model_outputs["attentions"]
            }

            return result

        return adaptor
