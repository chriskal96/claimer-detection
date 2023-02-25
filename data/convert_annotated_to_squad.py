import json

entry_list = []
i = 1
f = open(
    "/Users/christoskallaras/code/personal/claimer_detection/annotation/annotated_dataset.json"
)

data = json.load(f)
for d in data:
    title = d["title"]
    text = d["text"]
    question = d["question"]
    answer = d["answer"]
    answer_start = d["answer_start"]
    if answer == "Author":
        answer_start = -1
    entry_list.append(
        {
            "title": title,
            "paragraphs": [
                {
                    "qas": [
                        {
                            "question": question,
                            "id": i,
                            "answers": [{"text": answer, "answer_start": answer_start}],
                            "is_impossible": False,
                        }
                    ],
                    "context": text,
                }
            ],
        }
    )
    i = i + 1
final_string = {"version": "v2.0", "data": entry_list}
with open("annotated_to_squad.json", "w") as outfile:
    json.dump(final_string, outfile)
