import math
import torch
import os
import logging
from transformers import AutoTokenizer
from Dataset import DataIterator
from Model import GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from tensorboardX import SummaryWriter


# 初始化设备
device = torch.device("cuda:0")

# 得到special ids
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="")
space_id, content_id, title_id = tokenizer.convert_tokens_to_ids(["[unused1]", "[unused2]", "[unused3]"])

# 初始化数据集
data = DataIterator.from_corpus(
    corpus_tag_or_dir="",
    tokenizer_path="",
    max_title_length=30,
    content_id=content_id,
    title_id=title_id,
    space_id=space_id,
    batch_size=10,
    shuffle=True,
    device=device
)
train_iter, eval_iter, config = data["train_iter"], data["eval_iter"], data["config"]

# 初始化模型
model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="")
model.train()
model.to(device)

# 初始化训练参数
gradient_accumulation_steps = 4
max_grad_norm = 1.0
total_train_epochs = 10
save_per_n_epochs = 1
log_per_n_steps = 100

# 计算迭代参数
total_train_steps = math.ceil(config["train_size"] * total_train_epochs / config["batch_size"])
save_per_n_steps = math.ceil(config["train_size"] * save_per_n_epochs / config["batch_size"])
steps_one_epoch = math.ceil(config["train_size"] / config["batch_size"])

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_train_steps),
    num_training_steps=total_train_steps)

# 初始化保存路径
summaries_writer = SummaryWriter("")
model_save_path = ""


def train():
    # 训练
    iter_bar = tqdm(range(total_train_steps), desc="Train Steps (loss=X.XXX)", disable=False)
    accumulation_loss = 0.0

    for train_step in iter_bar:
        if train_step % steps_one_epoch == 0:
            logging.info(f"start {int(train_step / steps_one_epoch) + 1}th training epoch.")
        elif train_step > 0 and train_step % log_per_n_steps == 0:
            logging.info(f"progress to {train_step}th training step.")

        # 得到one batch数据
        train_data = next(train_iter)

        # forward
        train_output = model.forward(**train_data, title_id=title_id)
        train_loss = train_output[0]

        # 进行记录
        summaries_writer.add_scalar("train_loss", train_loss, train_step)
        iter_bar.set_description("Iter (loss=%5.3f)" % train_loss.item())

        # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
        if gradient_accumulation_steps > 1:
            train_loss = train_loss / gradient_accumulation_steps

        accumulation_loss += train_loss.item()

        # 损失进行回传
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 当训练步数整除累积步数时，进行参数优化
        if (train_step + 1) % gradient_accumulation_steps == 0 or train_step + 1 == total_train_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 如果步数整除log_per_n_steps * gradient_accumulation_steps，则记录学习率和训练集损失值
            if (train_step + 1) % (log_per_n_steps * gradient_accumulation_steps) == 0 or train_step + 1 == total_train_steps:
                logging.info(f"{int(train_step / steps_one_epoch + 1)}th epoch {train_step}th step training loss is {accumulation_loss}")

                summaries_writer.add_scalar("lr", scheduler.get_lr()[0], train_step)
                summaries_writer.add_scalar("train_loss", accumulation_loss, train_step)

            # 如果步数整除save_per_n_steps * gradient_accumulation_steps，则保存模型
            if (train_step + 1) % (save_per_n_steps * gradient_accumulation_steps) == 0:
                output_dir = os.path.join(model_save_path, "checkpoint-{}".format(train_step))
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)

            accumulation_loss = 0.0

        # 每个epoch进行完，则保存模型
        output_dir = os.path.join(model_save_path, "final")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)

        return


if __name__ == '__main__':
    train()
