from Summarization import Summarizer


summary_model_path = ""
summary_tokenizer_path = ""

agent = Summarizer(summary_model_path, summary_tokenizer_path)
content = "澳门2014年现金分享计划将于7月2日正式实施。届时，澳门特区永久性居民及非永久性居民将分别获发9000和5400澳门元。澳门特区政府此项财政开支约为56.59亿。为市民共享经济发展成果，澳门08年起推出现金分享计划。"  # input("输入的新闻正文为:")
print(f"原文: {content}")

summaries = agent.generate_summary(content, greedy=False)
for i, title in enumerate(summaries):
    print("生成的第{}个标题为：{}".format(i + 1, title))
