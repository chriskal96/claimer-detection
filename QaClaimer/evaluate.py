import argparse
import json
import logging
import os
import timeit

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult

import utils as utils

logger = logging.getLogger(__name__)


def _to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = utils.transform_data_to_features(args, tokenizer, evaluate=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.batches_per_gpu_eval * max(1, args.number_gpu)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Include parallel train if multi gpu exists
    if args.number_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Evaluate
    logger.info('***** Evaluation *****')
    logger.info(f'Num examples = {len(dataset)}')
    logger.info(f'Batch size = {args.eval_batch_size}')

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            feature_indices = batch[3]

            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [_to_list(output[i]) for output in outputs]

            # Bart use 3 arguments for its predictions, while the other models only use two.
            if len(output) >= 3:
                start_logits = output[0]
                end_logits = output[1]
                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits
                )
            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        True,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def main():
    parser = argparse.ArgumentParser()

    args = utils.parse_args(parser)

    # Setup CUDA and GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.number_gpu = torch.cuda.device_count() if args.device == "cuda" else 0

    # Setup logging
    logging.root.setLevel(logging.NOTSET)

    # Evaluation
    results = {}
    logger.info("Loading checkpoint for evaluation")
    checkpoints = [args.model_name_or_path]

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)
        model.to(args.device)

        # Evaluate
        result = evaluate(args, model, tokenizer, prefix=global_step)

        result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
        results.update(result)

    logger.info("Results: {}".format(results))
    with open('results.txt', 'w') as convert_file:
        convert_file.write(json.dumps(results))
    return results


if __name__ == "__main__":
    main()
