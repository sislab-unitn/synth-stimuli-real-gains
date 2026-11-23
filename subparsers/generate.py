import argparse
import json
import os
import random
import gc
from functools import partial
from typing import List, Optional


from peft import PeftModel
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, GenerationConfig

from utils.utils import MAP_MODELS_TRAIN, SUREDataset, MAP_MODELS


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


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser( "generate", help="Generate text from a model on the test set of a specific dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--instruction", metavar="INSTR", type=str, default="", help="Instruction used for generation.")
    parser.add_argument("--max-new-tokens", metavar="N_TOKENS", type=int, default=15, help="Maximum number of new tokens to generate.")
    parser.add_argument("--batch-size", metavar="BATCH_SIZE", type=int, default=2, help="Batch size for generation.")
    parser.add_argument("--debug", action="store_true", help="Use only 100 questions to check the answer of the model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to lora weights")
    parser.add_argument("--split", type=str, default=None, help="Data split to select.")

    parser.set_defaults(func=main)

def main(args):
    generate(
        model_name=args.model_name,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        debug=args.debug,
        model_path=args.model_path,
        split=args.split,
        parallel=args.parallel,
        device=args.device,
        seed=args.seed,
        out_dir=args.out_dir,
        data_folder=args.data_folder
    )

def generate(
    model_name: str,
    data_folder: str,
    instruction: str = "",
    model_path: Optional[str] = None,
    split: Optional[str] = None,
    batch_size: int = 2,
    out_dir: str = "output",
    max_new_tokens: int = 15,
    parallel: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    debug: bool = False,
    select_q_types: Optional[List[str]] = None
):
    n_stim = None
    if debug:
        n_stim = 10

    assert model_name != "CLIP", "This script is for text generation models only."
    hf_model_name, Model, Collator = MAP_MODELS[model_name]

    top_k = 0
    top_p = 1.0
    temperature = 1.0

    base_model = Model(
        hf_model_name,
        low_cpu_mem_usage=True,
        device_map="auto" if parallel else device,
    )
    model = base_model
    if model_path is not None:
        print(f"Loading LoRa adapters from {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()

    Processor = partial(AutoProcessor.from_pretrained, hf_model_name)
    if model_name == "qwen2-vl-7b":
        # as declared in the paper
        processor = Processor(
            min_pixels=100 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        )
    elif model_name == "molmo-o-7b":
        processor = Processor(
            trust_remote_code=True,
        )
    else:
        processor = Processor()

    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id  # type: ignore
    processor.tokenizer.padding_side = "left"  # type: ignore

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load ids for already generated samples
    generated_samples = []
    if os.path.exists(os.path.join(out_dir, f"generated_samples.json")):
        with open(os.path.join(out_dir, f"generated_samples.json"), "r") as f:
            generated_samples = json.load(f)

    # load ids of split (if specified)
    select_ids = None
    split_path = os.path.join(data_folder, "split.json")
    if split is not None and os.path.exists(split_path):
        print("Loading train/val splits...")
        with open(split_path, "r") as f:
            splits_tmp = json.load(f)
        splits = {split_name: [tuple(id_elems) for id_elems in split] for split_name, split in splits_tmp.items()}
        print(f"Selecting {split} ids of '{data_folder}'")
        select_ids = splits[split]

    # load test data
    test_ds = SUREDataset(data_folder, generated_samples, select_q_types=select_q_types, select_ids=select_ids, n_stim=n_stim)
    test_loader = DataLoader(
        test_ds,
        batch_size=1 if model_name == "molmo-o-7b" else batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(
            processor=processor,  # type: ignore
            instruction=instruction,
        ),
    )


    os.makedirs(out_dir, exist_ok=True)

    generation_results = {}
    if os.path.exists(os.path.join(out_dir, f"generation_results.json")):
        with open(os.path.join(out_dir, f"generation_results.json"), "r") as f:
            generation_results = json.load(f)

    # generate
    with torch.no_grad():
        for input_ids, _, targets, sample_ids in tqdm(
            test_loader, desc=f"Generating with P={top_p}, T={temperature}, K={top_k}"
        ):
            # add the generated samples to keep track of them
            generated_samples.extend(sample_ids)

            if model_name == "molmo-o-7b":
                input_ids = {
                    k: v.to(model.device).unsqueeze(0) for k, v in input_ids.items()
                }

                input_ids["images"] = input_ids["images"].to(torch.bfloat16)

                output = model.generate_from_batch(
                    input_ids,
                    GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        stop_strings="<|endoftext|>",  # suggested on MOLMo huggingface page
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                    ),
                    tokenizer=processor.tokenizer,
                )
                generated_tokens = output[0, input_ids["input_ids"].size(1) :]
                generated_answers = [
                    processor.tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    )
                ]
            else:
                input_ids.to(model.device)
                output = model.generate(
                    **input_ids,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )

                if "llava-next" in model_name:
                    output = output[:, input_ids.input_ids.size(-1) :]
                elif model_name == "qwen2-vl-7b":
                    output = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(input_ids.input_ids, output)
                    ]
                elif model_name == "llava-onevision-7b":
                    output = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(input_ids.input_ids, output)
                    ]

                generated_answers = processor.batch_decode(  # type: ignore
                    output, skip_special_tokens=True
                )

            for sample_id, pred, target in zip(sample_ids, generated_answers, targets):
                img_id, ent_id, *q_type = sample_id.split("-")

                if img_id not in generation_results:
                    generation_results[img_id] = {}

                if ent_id not in generation_results[img_id]:
                    generation_results[img_id][ent_id] = {}

                if len(q_type) == 2:
                    q_type, sub_q_type = q_type
                    if q_type not in generation_results[img_id][ent_id]:
                        generation_results[img_id][ent_id][q_type] = {}

                    generation_results[img_id][ent_id][q_type][sub_q_type] = {
                        "target": target,
                        "pred": pred,
                    }
                elif len(q_type) == 1:
                    q_type = q_type[0]
                    generation_results[img_id][ent_id][q_type] = {
                        "target": target,
                        "pred": pred,
                    }
                else:
                    raise ValueError(f"Invalid question type: {q_type}")

            with open(os.path.join(out_dir, f"generation_results.json"), "w") as f:
                json.dump(generation_results, f, indent=4)

            with open(os.path.join(out_dir, f"generated_samples.json"), "w") as f:
                json.dump(generated_samples, f, indent=4)

    del model
    torch.cuda.empty_cache()
    gc.collect()


def eval_clip(
    model_name: str,
    data_folder: str,
    instruction: str = "",
    model_path: Optional[str] = None,
    split: Optional[str] = None,
    out_dir: str = "output",
    max_new_tokens: int = 15,
    parallel: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
    debug: bool = False,
    select_q_types: Optional[List[str]] = None
):
    n_stim = None
    if debug:
        n_stim = 10

    hf_model_name, Model, Collator, _ = MAP_MODELS_TRAIN["CLIP"]

    base_model = Model.from_pretrained(
        hf_model_name,
        low_cpu_mem_usage=True,
        device_map="auto" if parallel else device,
    )
    model = base_model
    if model_path is not None:
        print(f"Loading LoRa adapters from {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()

    Processor = partial(AutoProcessor.from_pretrained, hf_model_name)
    processor = Processor(
        do_resize=True,
        input_data_format="channels_last",
    )

    collator = Collator(
        processor=processor,  # type: ignore
        candidates=list(AREA_MAPS["position_absolute"].values())
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load ids for already generated samples
    generated_samples = []
    if os.path.exists(os.path.join(out_dir, f"generated_samples.json")):
        with open(os.path.join(out_dir, f"generated_samples.json"), "r") as f:
            generated_samples = json.load(f)

    # load ids of split (if specified)
    select_ids = None
    split_path = os.path.join(data_folder, "split.json")
    if split is not None and os.path.exists(split_path):
        print("Loading train/val splits...")
        with open(split_path, "r") as f:
            splits_tmp = json.load(f)
        splits = {split_name: [tuple(id_elems) for id_elems in split] for split_name, split in splits_tmp.items()}
        print(f"Selecting {split} ids of '{data_folder}'")
        select_ids = splits[split]

    # load test data
    test_ds = SUREDataset(data_folder, generated_samples, select_q_types=select_q_types, select_ids=select_ids, n_stim=n_stim)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator
    )

    os.makedirs(out_dir, exist_ok=True)

    generation_results = {}
    if os.path.exists(os.path.join(out_dir, f"generation_results.json")):
        with open(os.path.join(out_dir, f"generation_results.json"), "r") as f:
            generation_results = json.load(f)

    # generate
    with torch.no_grad():
        for input_ids, text_candidates, target, target_id, sample_id in tqdm(
            test_loader, desc=f"Computing predictions for CLIP"
        ):
            # add the generated samples to keep track of them
            generated_samples.append(sample_id)

            input_ids.to(model.device)
            output = model(**input_ids)

            # get answer by selecting by highest similarity / probability
            answ_index = output.logits_per_image.softmax(dim=1).max(dim=1).indices[0].item()
            answer_candidate = text_candidates[answ_index]
            answer = collator.candidates[answ_index]
            assert answer in answer_candidate, f"'{answer}' not in '{answer_candidate}'."
            assert answer in processor.tokenizer.decode(input_ids.input_ids[answ_index])
            assert target in processor.tokenizer.decode(input_ids.input_ids[target_id])

            img_id, ent_id, *q_type = sample_id.split("-")

            if img_id not in generation_results:
                generation_results[img_id] = {}

            if ent_id not in generation_results[img_id]:
                generation_results[img_id][ent_id] = {}

            if len(q_type) == 2:
                q_type, sub_q_type = q_type
                if q_type not in generation_results[img_id][ent_id]:
                    generation_results[img_id][ent_id][q_type] = {}

                generation_results[img_id][ent_id][q_type][sub_q_type] = {
                    "target": target,
                    "pred": answer,
                }
            elif len(q_type) == 1:
                # unpack it
                q_type = q_type[0]
                generation_results[img_id][ent_id][q_type] = {
                    "target": target,
                    "pred": answer,
                }
            else:
                raise ValueError(f"Invalid question type: {q_type}")


            with open(os.path.join(out_dir, f"generation_results.json"), "w") as f:
                json.dump(generation_results, f, indent=4)

            with open(os.path.join(out_dir, f"generated_samples.json"), "w") as f:
                json.dump(generated_samples, f, indent=4)

    del model
    torch.cuda.empty_cache()
    gc.collect()
