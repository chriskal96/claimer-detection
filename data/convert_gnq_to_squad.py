import json


def transform_gnq(file: str, train: bool):
    entry_list = []
    j = 0
    no_short = 0
    no_long = 0
    with open(file) as f:
        for line in f:
            infile = json.loads(line)
            is_impossible = False
            if not infile["annotations"][0]["short_answers"]:
                no_short += 1
                is_impossible = True

            if infile["annotations"][0]["long_answer"]["start_token"] == -1:
                no_long += 1
                continue
            text = infile["document_html"]
            question = infile["question_text"]
            question_id = infile["example_id"]
            la_start_token = infile["annotations"][0]["long_answer"]["start_token"]
            la_end_token = infile["annotations"][0]["long_answer"]["end_token"]
            tok_ans = ""
            ans_idx = 0
            text_no_html = ""
            prev_token = ""
            h1start = False
            title = ""
            for i in range(len(infile["document_tokens"])):
                token = infile["document_tokens"][i]
                if token["token"] == "<H1>":
                    h1start = True
                if token["token"] == "</H1>" and title != "":
                    h1start = False
                    break
                if not token["html_token"]:
                    if (
                        prev_token != ""
                        and prev_token != "("
                        and prev_token != "``"
                        and prev_token != "'"
                        and prev_token != "-"
                        and prev_token != "--"
                        and (
                            token["token"] == "("
                            or token["token"] == "``"
                            or token["token"] == "'"
                            or (token["token"].replace(".", "").isalnum())
                        )
                    ):
                        if h1start:
                            if title == "":
                                title += token["token"]
                            else:
                                title += " " + token["token"]
                    else:
                        if h1start:
                            if title == "":
                                title += token["token"]
                            else:
                                title += token["token"]
                    prev_token = token["token"]
            prev_token = ""

            if is_impossible:
                for i in range(la_start_token, la_end_token):
                    token = infile["document_tokens"][i]
                    if not token["html_token"]:
                        if (
                            prev_token != ""
                            and prev_token != "("
                            and prev_token != "``"
                            and prev_token != "'"
                            and prev_token != "-"
                            and prev_token != "--"
                            and (
                                token["token"] == "("
                                or token["token"] == "``"
                                or token["token"] == "'"
                                or (token["token"].replace(".", "").isalnum())
                            )
                        ):
                            text_no_html += " " + token["token"]
                        else:
                            text_no_html += token["token"]
                        prev_token = token["token"]

            else:
                for i in range(la_start_token, la_end_token):
                    token = infile["document_tokens"][i]
                    before_ans = False
                    in_ans = False
                    overlap = False

                    if i < infile["annotations"][0]["short_answers"][0]["start_token"]:
                        before_ans = True
                    elif (
                        infile["annotations"][0]["short_answers"][0]["start_token"]
                        <= i
                        < infile["annotations"][0]["short_answers"][0]["end_token"]
                    ):
                        in_ans = True
                    if i == infile["annotations"][0]["short_answers"][0]["start_token"]:
                        overlap = True

                    if not token["html_token"]:
                        if (
                            prev_token != ""
                            and prev_token != "("
                            and prev_token != "``"
                            and prev_token != "'"
                            and prev_token != "-"
                            and prev_token != "--"
                            and (
                                token["token"] == "("
                                or token["token"] == "``"
                                or token["token"] == "'"
                                or (token["token"].replace(".", "").isalnum())
                            )
                        ):
                            text_no_html += " " + token["token"]
                            if overlap:
                                ans_idx += 1
                                tok_ans += token["token"]
                            elif before_ans:
                                ans_idx += len(token["token"]) + 1
                            elif in_ans:
                                tok_ans += " " + token["token"]

                        else:
                            text_no_html += token["token"]
                            if before_ans:
                                ans_idx += len(token["token"])
                            elif in_ans:
                                tok_ans += token["token"]

                        prev_token = token["token"]
            if is_impossible:
                entry_list.append(
                    {
                        "title": title,
                        "paragraphs": [
                            {
                                "qas": [
                                    {
                                        "plausible_answers": [{}],
                                        "question": question,
                                        "id": question_id,
                                        "answers": [],
                                        "is_impossible": is_impossible,
                                    }
                                ],
                                "context": text_no_html,
                            }
                        ],
                    }
                )
            else:
                entry_list.append(
                    {
                        "title": title,
                        "paragraphs": [
                            {
                                "qas": [
                                    {
                                        "question": question,
                                        "id": question_id,
                                        "answers": [
                                            {"text": tok_ans, "answer_start": ans_idx}
                                        ],
                                        "is_impossible": is_impossible,
                                    }
                                ],
                                "context": text_no_html,
                            }
                        ],
                    }
                )
            j += 1

    final_string = {"version": "v2.0", "data": entry_list}
    if train:
        with open("gnq_squad_form_train.json", "w") as outfile:
            json.dump(final_string, outfile)
    else:
        with open("gnq_squad_form_eval.json", "w") as outfile:
            json.dump(final_string, outfile)


transform_gnq(
    "/Users/christoskallaras/code/personal/claimer_detection/natural_question/v1.0-simplified_simplified-nq-train.jsonl"
)
transform_gnq(
    "/Users/christoskallaras/code/personal/claimer_detection/natural_question/v1.0-simplified_nq-dev-all.jsonl"
)
