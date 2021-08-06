import math
import os
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from Dataset import DataIterator
from tensorboardX import SummaryWriter
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import logging


# 初始化设备
device = torch.device("cuda:0")

# 初始化数据集
data = DataIterator.from_corpus(
    corpus_tag_or_dir="",
    tokenizer_path="",
    batch_size=10,
    shuffle=True,
    device=device
)
train_iter, eval_iter, config = data["train_iter"], data["eval_iter"], data["config"]

# 初始化模型
model = AutoModelForSequenceClassification.from_pretrained("", num_labels=10)
model.train()
model.to(device)

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
grad_scaler = GradScaler()

# 初始化保存路径
summaries_writer = SummaryWriter("")
model_save_path = ""

# 初始化训练参数
total_train_epochs = 10
total_eval_epochs = 1
eval_per_n_epochs = 1
save_per_n_epochs = 1
log_per_n_steps = 100


def train_with_eval():
    # 计算迭代参数
    total_train_steps = math.ceil(config["train_size"] * total_train_epochs / config["batch_size"])
    total_eval_steps = math.ceil(config["eval_size"] * total_eval_epochs / config["batch_size"])
    eval_per_n_steps = math.ceil(config["train_size"] * eval_per_n_epochs / config["batch_size"])
    save_per_n_steps = math.ceil(config["train_size"] * eval_per_n_epochs / config["batch_size"])
    steps_one_epoch = math.ceil(config["train_size"] / config["batch_size"])

    # 训练
    for train_step in range(total_train_steps):
        if train_step % steps_one_epoch == 0:
            logging.info(f"start {int(train_step / steps_one_epoch) + 1}th training epoch.")
        elif train_step > 0 and train_step % log_per_n_steps == 0:
            logging.info(f"progress to {train_step}th training step.")

        # 得到one batch数据
        train_data = next(train_iter)

        # forward
        with autocast():
            train_output = model(**train_data)
            train_loss = train_output["loss"]

        logging.info(f"{int(train_step / steps_one_epoch + 1)}th epoch {train_step}th step training loss is {train_loss}")
        summaries_writer.add_scalar("train_loss", train_loss, train_step)

        # backward
        optimizer.zero_grad()
        grad_scaler.scale(train_loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # 进行测试
        if eval_iter is not None:
            if train_step > 0 and train_step % eval_per_n_steps == 0:
                logging.info(f"start {int(train_step / eval_per_n_steps)}th model evaluation.")

                # 初始化
                model.eval()
                predictions = []
                labels = []

                for eval_step in range(total_eval_steps):
                    # 得到one batch数据
                    eval_data = next(eval_iter)
                    batch_labels = eval_data["labels"]
                    labels += batch_labels.detach().cpu().numpy().tolist()

                    # forward
                    eval_output = model(**eval_data)
                    batch_predictions = torch.argmax(eval_output.logits, dim=-1).detach().cpu().numpy().tolist()
                    predictions += batch_predictions

                # 计算f1 score
                f1 = f1_score(labels, predictions)
                logging.info(f"{int(train_step / eval_per_n_steps)}th epoch f1 score is {f1}")
                summaries_writer.add_scalar("eval f1", f1, int(train_step / eval_per_n_steps))
                logging.info(f"{int(train_step / eval_per_n_steps)}th model evaluation is completed.")

                model.train()

        # 进行保存
        if train_step > 0 and train_step % save_per_n_steps == 0:
            logging.info(f"start {int(train_step / save_per_n_steps)}th model saving.")

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            model_save_name = os.path.join(model_save_path, f"pytorch_model_{train_step}.h5")
            torch.save(model, model_save_name)

            logging.info(f"{int(train_step / save_per_n_steps)}th model saving is completed.")

        if (train_step + 1) % steps_one_epoch == 0:
            logging.info(f"{int(train_step / steps_one_epoch) + 1}th training epoch is completed.")

    return


if __name__ == '__main__':
    train_with_eval()
