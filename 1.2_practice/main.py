from Distiller import Distiller


def main():
    distiller = Distiller()
    task_tag = "L3TC"
    corpus_tag_or_path = "/home/ubuntu/Desktop/citic_corpus/preprocessed_corpus/aspect"
    teacher_model = "/home/ubuntu/PycharmProjects/nlptoolkit/nlptoolkit/Schedule/checkpoint/bert_cn_continuance.h5"
    tokenizer_path = ""

    distiller.distill(
        task_tag=task_tag,
        corpus_tag_or_path=corpus_tag_or_path,
        teacher_model_or_path=teacher_model,
        tokenizer_path=tokenizer_path,
        batch_size=1,
    )


if __name__ == '__main__':
    main()
