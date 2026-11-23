import argparse
from functools import partial
import gc
import json
import os
import random
from pathlib import Path
from tqdm import tqdm


from subparsers.generate import eval_clip, generate
import torch
import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoProcessor


from sklearn.model_selection import train_test_split

from utils.utils import (
    MODEL_TO_NAME,
    Checkpointer,
    MAP_MODELS_TRAIN,
    SUREDataset,
    load_training_params,
    resume_training,
    save_training_params,
    seed_worker,
    train,
)

from world.world import ObjEntity, World


AREA_MAPS = {
    "position_absolute": {
        "NW": "top left",
        "N": "top center",
        "NE": "top right",
        "W": "center left",
        "C": "center",
        "E": "center right",
        "SW": "bottom left",
        "S": "bottom center",
        "SE": "bottom right"
    }
}


def create_splits(questions, repr, select_q_types, args):
    all_ids = []
    all_labels = []
    for w_id, ents in questions.items():
        w_repr = repr[w_id]
        world = World.from_repr(w_repr)
        for e_id, questions in ents.items():
            ent = world.ents[e_id]
            for q_type in questions.keys():
                if q_type in select_q_types:
                    all_ids.append((w_id, e_id, q_type))
                    if isinstance(ent, ObjEntity):
                        all_labels.append((ent.area, ent.color, ent.shape, ent.size))
                    else:
                        all_labels.append((ent.area, ent.category))

    train_ids, val_ids, train_labels, val_labels = train_test_split(all_ids, all_labels, test_size=0.2, stratify=all_labels)
    print(len(all_ids), len(train_ids), len(val_ids))

    splits = {
        "train": train_ids,
        "val": val_ids
    }

    return splits


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "fine-tune",
        help="Fine-Tune a model on a specific dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--epochs", metavar="EPOCHS", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--save-every", metavar="SAVE_EVERY", type=int, default=100, help="Save model every SAVE_EVERY steps.")
    parser.add_argument("--lr", metavar="LR", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--max-patience", metavar="MAX_PATIENCE", type=int, default=2, help="Maximum patience for early stopping.")
    parser.add_argument("-r", "--rank", metavar="R", type=int, default=32, help="Rank for LoRA.")
    parser.add_argument("--lora-alpha", metavar="LORA_ALPHA", type=int, default=64, help="Alpha value for LoRA.")
    parser.add_argument("--batch-size", metavar="BATCH_SIZE", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--val-batch-size", metavar="VAL_BATCH_SIZE", type=int, default=16, help="Batch size for validation.")

    parser.add_argument("--instruction", metavar="INSTR", type=str, default="Answer using as few words as possible.", help="Instruction used for generation.")

    parser.add_argument("--test", action="store_true", help="Perform a test on some samples.")
    parser.add_argument("--n_stim", type=int, default=None, help="Number of stimuli to sample from data.")
    parser.add_argument("--n_stim_val", type=int, default=None, help="Number of stimuli to sample from data for the validation set.")

    parser.add_argument("--test_sets", type=str, nargs="*", default=None, help="Paths to test sets.")

    parser.add_argument("--runs", type=int, default=1, help="Number of runs")


    parser.add_argument("--test_only", action="store_true", help="Do not fine-tune. Only compute performance on the selected test sets")


    parser.set_defaults(func=main)


def main(args):
    del args.func
    test = args.test
    del args.test

    if test and args.n_stim is None:
        args.n_stim = 3

    if args.n_stim is not None and args.n_stim_val is None:
        args.n_stim_val = args.n_stim
    print(f"Train n_stim: {args.n_stim}")
    print(f"Valid n_stim: {args.n_stim_val}")


    model_name, Model, Collator, lora_targets = MAP_MODELS_TRAIN[args.model_name]

    for i in tqdm(range(args.runs), unit="run", desc="Executing runs"):
        seed = args.seed
        if args.runs > 1:
            seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        Processor = partial(AutoProcessor.from_pretrained, model_name)
        if args.model_name == "qwen2-vl-7b":
            # as declared in the paper
            processor = Processor(
                min_pixels=100 * 28 * 28,
                max_pixels=16384 * 28 * 28,
            )
        elif args.model_name == "molmo-o-7b":
            processor = Processor(
                trust_remote_code=True,
            )
        elif args.model_name == "CLIP":
            processor = Processor(
                do_resize=True,
                input_data_format="channels_last",
            )
        else:
            processor = Processor()


        output_folder = args.out_dir
        if args.runs > 1:
            output_folder = os.path.join(output_folder, f"run_{i}")

        if not os.path.exists(os.path.join(output_folder, "training_args.json")):
            os.makedirs(output_folder, exist_ok=True)
            save_training_params(args, output_folder)
        else:
            print(f"Experiment '{args.experiment_name}' already exists.")
            print(f"Loading training parameters from {output_folder}.")
            args = load_training_params(output_folder)

        select_q_types = ["position_absolute"]

        if not args.test_only:
            criterion = CrossEntropyLoss(ignore_index=-100, reduction="sum")

            if args.model_name != "CLIP":
                model = Model(
                    model_name,
                    device_map="auto" if args.parallel else args.device,
                )
            else:
                model = Model.from_pretrained(
                    model_name,
                    device_map="auto" if args.parallel else args.device,
                )

            checkpointer = Checkpointer(args)
            early_stopping = checkpointer.checkpoint.early_stopping
            if os.path.exists(os.path.join(output_folder, "checkpoint")):
                if "cuda" in args.device:
                    checkpoint, model = checkpointer.load_checkpoint(model, output_folder)
                else:
                    assert args.device == "cpu"
                    checkpoint, model = checkpointer.load_checkpoint(model, output_folder, args.device)
                early_stopping = checkpoint.early_stopping
                start_epoch = checkpoint.epoch
            else:
                start_epoch = 0

                config = LoraConfig(
                    r=args.rank,
                    lora_alpha=args.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=lora_targets
                )

                model = get_peft_model(model, config)

            model.print_trainable_parameters()

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # type: ignore

            if checkpointer.checkpoint.optimizer is not None:
                optimizer.load_state_dict(checkpointer.checkpoint.optimizer)

            split_path = os.path.join(args.data_folder, "split.json")
            if os.path.exists(split_path):
                print("Loading train/val splits...")
                with open(split_path, "r") as f:
                    splits_tmp = json.load(f)
                splits = {split_name: [tuple(id_elems) for id_elems in split] for split_name, split in splits_tmp.items()}
            else:
                print("Creating train/val splits...")
                with open(os.path.join(args.data_folder, "questions.json")) as f:
                    questions = json.load(f)
                with open(os.path.join(args.data_folder, "repr.json")) as f:
                    repr = json.load(f)

                splits = create_splits(questions, repr, select_q_types, args)

                with open(split_path, "w") as f:
                    json.dump(splits, f, indent=4)

            train_ds = SUREDataset(args.data_folder, [], select_q_types=select_q_types, select_ids=splits["train"], n_stim=args.n_stim)
            valid_ds = SUREDataset(args.data_folder, [], select_q_types=select_q_types, select_ids=splits["val"], n_stim=args.n_stim_val)

            if args.model_name != "CLIP":
                collator = Collator(
                    processor=processor,  # type: ignore
                    instruction=args.instruction,
                )
            else:
                collator = Collator(
                    processor=processor,  # type: ignore
                    candidates=list(AREA_MAPS["position_absolute"].values())
                )

            train_loader = DataLoader(
                train_ds,
                batch_size=1 if args.model_name == "molmo-o-7b" else args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                collate_fn=collator,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed),
            )
            valid_loader = DataLoader(
                valid_ds,
                batch_size=1 if args.model_name == "molmo-o-7b" else args.val_batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=collator
            )

            if resume_training(start_epoch, args, early_stopping):
                train(
                    args,
                    model,  # type: ignore
                    train_loader,  # type: ignore
                    valid_loader,
                    criterion,
                    optimizer,
                    output_folder,
                    checkpointer,
                    processor.tokenizer,
                    test=test,
                    clip=(args.model_name == "CLIP")
                )

            del model
            torch.cuda.empty_cache()
            gc.collect()

        if args.test_sets is not None:
            for test_set_path in args.test_sets:
                print(f"Test on '{test_set_path}'")
                if args.model_name != "CLIP":
                    generate(
                        args.model_name,
                        test_set_path,
                        args.instruction,
                        out_dir=os.path.join(output_folder, Path(test_set_path).name),
                        model_path=os.path.join(output_folder, "best_model") if not args.test_only else None, # no model weights if --test_only
                        batch_size=args.val_batch_size,
                        parallel=args.parallel,
                        device=args.device,
                        seed=seed,
                        debug=test,
                        select_q_types=select_q_types
                    )
                else:
                    # CLIP
                    eval_clip(
                        args.model_name,
                        test_set_path,
                        args.instruction,
                        out_dir=os.path.join(output_folder, Path(test_set_path).name),
                        model_path=os.path.join(output_folder, "best_model") if not args.test_only else None, # no model weights if --test_only
                        parallel=args.parallel,
                        device=args.device,
                        seed=seed,
                        debug=test,
                        select_q_types=select_q_types
                    )
