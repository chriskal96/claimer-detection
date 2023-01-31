import argparse
import logging
from dataclasses import dataclass

import spacy
from transformers import pipeline

from QaClaimer import utils


@dataclass
class Prediction:
    context: str = None
    question: str = None
    answer: str = None


def predict(context, claim, model, tokenizer, threshold=0.01):
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    question = 'Who said that ' + claim + '?'

    qa_input = {
        'question': question,
        'context': context
    }
    res = nlp(qa_input)

    nlp_spacy = spacy.load("en_core_web_sm")

    doc = nlp_spacy(res['answer'])

    if len(doc.ents) == 0:
        res['score'] = 0
    else:
        for ent in doc.ents:
            if ent.label_ not in ('ORG', 'PERSON'):
                res['score'] = 0

    if res['score'] > threshold:
        answer = res['answer']
    else:
        answer = 'Author'

    return Prediction(
        context=context,
        question=question,
        answer=answer,
    )


def main():
    # Setup logging
    logging.root.setLevel(logging.NOTSET)

    # Parse input arguments
    parser = argparse.ArgumentParser()

    args = utils.parse_args(parser)

    prediction = predict(args.context, args.claim, args.model_name_or_path, args.model_name_or_path, args.threshold)

    print(f"\nquestion: {prediction.question}\n")
    print(f"context: {prediction.context}")
    print(f"\nanswer: {prediction.answer}\n")

    return 0


if __name__ == "__main__":
    main()
