import argparse
import json
import logging
import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import utils as utils
from evaluate import evaluate

logger = logging.getLogger(__name__)


def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.batches_per_gpu_train * max(1, args.number_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # in case of multi-gpu training
    if args.number_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Training *****")

    global_step = 1
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
    )

    utils.set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)

            loss = outputs[0]

            if args.number_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.evaluate_during_training:
                    results = evaluate(args, model, tokenizer)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

                # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def main():
    # Setup logging
    logging.root.setLevel(logging.NOTSET)

    # Parse input arguments
    parser = argparse.ArgumentParser()

    args = utils.parse_args(parser)

    # Setup CUDA and GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.number_gpu = torch.cuda.device_count() if args.device == "cuda" else 0

    # Set seed
    utils.set_seed(args)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(args.model_name_or_path, )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=args.do_lower_case,
                                              use_fast=False,
                                              )

    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    model.to(args.device)

    # Training
    train_dataset = utils.transform_data_to_features(args, tokenizer, evaluate=False)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)

    # Save the trained model and the tokenizer
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load the trained model and vocabulary
    model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)
    model.to(args.device)

    # Evaluation
    logger.info("Evaluation after Training")
    results = {}

    logger.info("Load checkpoints saved during training")
    checkpoints = [args.output_dir]

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
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
