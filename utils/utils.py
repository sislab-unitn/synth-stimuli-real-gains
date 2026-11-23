import json
import math
import os
import random
from argparse import Namespace
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Tuple
from statistics import mean

import numpy as np
import torch
from peft.peft_model import PeftModel
from PIL import Image
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    LlavaNextForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from utils import molmo
from utils.llava_next import GenerationCollator as LLavaNextGenerationCollator
from utils.molmo import GenerationCollator as MOLMoGenerationCollator
from utils.qwen2_vl import GenerationCollator as Qwen2VLGenerationCollator
from utils.llava_onevision import GenerationCollator as LlavaOnevisionGenerationCollator
from utils.clip import SURECollator
from utils import llava_next, qwen2_vl, llava_onevision
from world.world import World


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


MAP_MODELS_TRAIN = {
    "molmo-o-7b": (
        "allenai/Molmo-7B-O-0924",
        partial(
            AutoModelForCausalLM.from_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ),
        molmo.TrainCollator,
        ["wq", "wk", "wv", "att_proj"]
    ),
    "llava-next-7b": (
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        partial(
            LlavaNextForConditionalGeneration.from_pretrained,
            torch_dtype=torch.float16,
        ),
        llava_next.TrainCollator,
        ["q_proj", "k_proj", "v_proj"]
    ),
    "llava-onevision-7b": ( # needs at least transformers==4.53.0 (was transformers 4.46.3)
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        partial(
            AutoModelForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ),
        llava_onevision.TrainCollator,
        ["q_proj", "k_proj", "v_proj", "qkv"]
    ),
    "qwen2-vl-7b": (
        "Qwen/Qwen2-VL-7B-Instruct",
        partial(
            Qwen2VLForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        qwen2_vl.TrainCollator,
        ["q_proj", "k_proj", "v_proj", "qkv"]
    ),
    "CLIP": (
        "openai/clip-vit-large-patch14-336",
        CLIPModel,
        SURECollator,
        ["q_proj", "k_proj", "v_proj"]
    ),
}

MAP_MODELS = {
    "molmo-o-7b": (
        "allenai/Molmo-7B-O-0924",
        partial(
            AutoModelForCausalLM.from_pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ),
        MOLMoGenerationCollator,
    ),
    "llava-next-7b": (
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        partial(
            LlavaNextForConditionalGeneration.from_pretrained,
            torch_dtype=torch.float16,
        ),
        LLavaNextGenerationCollator,
    ),
    "llava-onevision-7b": (
        "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        partial(
            AutoModelForCausalLM.from_pretrained,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ),
        LlavaOnevisionGenerationCollator,
    ),
    "qwen2-vl-7b": (
        "Qwen/Qwen2-VL-7B-Instruct",
        partial(
            Qwen2VLForConditionalGeneration.from_pretrained,
            torch_dtype=torch.bfloat16,
        ),
        Qwen2VLGenerationCollator,
    ),
    "CLIP": (
        "openai/clip-vit-large-patch14-336",
        CLIPModel,
        SURECollator,
    ),
}

MODEL_TO_NAME = {
    model_params[0]: name
    for name, model_params in MAP_MODELS.items()
}


class SUREDataset(Dataset):
    def __init__(self, data_folder: str, generated_samples, select_q_types: Optional[List[str]] = None, select_ids: Optional[List[tuple]] = None, n_stim: Optional[int] = None):
        self.questions = []
        self.sample_ids = []
        self.answers = []
        self.images = []

        with open(os.path.join(data_folder, "questions.json"), "r") as f:
            data: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

        with open(os.path.join(data_folder, "repr.json"), "r") as f:
            repr = json.load(f)

        for img_id, entities in data.items():
            for ent_id, question_types in entities.items():
                for q_type, questions in question_types.items():
                    if select_q_types is not None and q_type not in select_q_types:
                        continue
                    if select_ids is not None and (img_id, ent_id, q_type) not in select_ids:
                        continue

                    answer = "error"
                    if q_type == "position_absolute":
                        world = World.from_repr(repr[img_id])
                        answer = AREA_MAPS[q_type][world.ents[ent_id].area]

                    if isinstance(questions, dict):
                        for sub_q_type, sub_question in questions.items():
                            key = f"{img_id}-{ent_id}-{q_type}-{sub_q_type}"
                            if key not in generated_samples:
                                self.questions.append(sub_question)
                                self.sample_ids.append(key)

                                self.answers.append(answer)

                                self.images.append(
                                    os.path.join(data_folder, "images", f"{img_id}.png")
                                )
                    else:
                        key = f"{img_id}-{ent_id}-{q_type}"
                        if key not in generated_samples:
                            self.questions.append(questions)
                            self.sample_ids.append(key)

                            self.answers.append(answer)

                            self.images.append(
                                os.path.join(data_folder, "images", f"{img_id}.png")
                            )

        if n_stim is not None:
            n_stim = min(n_stim, len(self.sample_ids))
            self.questions, self.sample_ids, self.answers, self.images = list(zip(
                *random.sample(
                    list(zip(self.questions, self.sample_ids, self.answers, self.images)),
                    n_stim
                )
            ))
            print(f"Sampled n_stim = {len(self.sample_ids)}")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, str, str, str]:
        return (
            Image.open(self.images[idx]),
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
        )


def compute_nll_and_ppl(
    losses: List[float], unmasked_tokens: int
) -> Tuple[float, float]:
    nll = sum(losses) / unmasked_tokens
    ppl = math.exp(nll)
    return nll, ppl


def evaluate(
    args,
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    is_test: bool = False,
    tokenizer: AutoTokenizer = None
) -> Tuple[float, float]:

    losses = []
    model.eval()  #  type: ignore
    unmasked_tokens = 0
    with torch.no_grad():
        for input_ids, labels in (pbar := tqdm(dataloader, desc="Evaluating")):
            if not isinstance(input_ids, dict):
                input_ids = input_ids.to(args.device)
                in_ids = input_ids.input_ids
            else:
                # molmo processor returns a dict
                input_ids = {k:v.to(args.device) for k,v in input_ids.items()}
                in_ids = input_ids["input_ids"]
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(**input_ids) # type: ignore

                labels_shifted = torch.full(in_ids.shape, tokenizer.pad_token_id)

                labels_shifted[
                    labels_shifted == tokenizer.pad_token_id
                ] = -100

                labels_shifted[:, :-1] = labels[:, 1:].detach().clone()

                losses.append(criterion(outputs.logits.permute(0, 2, 1), labels_shifted.to(args.device)).item())
                unmasked_tokens += (labels_shifted != -100).sum().item()
                _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)


            pbar.set_postfix(
                {
                    f"{'Test' if is_test else 'Valid'} PPL": ppl,
                }
            )

    return compute_nll_and_ppl(losses, unmasked_tokens)

def evaluate_clip(
    args,
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    is_test: bool = False,
    tokenizer: AutoTokenizer = None
) -> Tuple[float, float]:

    losses = []
    model.eval()  #  type: ignore
    with torch.no_grad():
        for input_ids, text_candidates, target, target_id, sample_id in (pbar := tqdm(dataloader, desc="Evaluating")):
            input_ids = input_ids.to(args.device)

            with torch.no_grad():
                output = model(**input_ids)

                assert target in tokenizer.decode(input_ids.input_ids[target_id])
                loss = criterion(output.logits_per_image, torch.tensor([target_id]).to(args.device))

                losses.append(loss.item())

            pbar.set_postfix(
                {
                    f"{'Test' if is_test else 'Valid'} loss": mean(losses),
                }
            )

    return mean(losses)



def save_training_params(args: Namespace, output_folder: str) -> None:
    with open(os.path.join(output_folder, "training_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def load_training_params(output_folder: str) -> Namespace:
    with open(os.path.join(output_folder, "training_params.json"), "r") as f:
        return Namespace(**json.load(f))


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_ppl = float("inf")
        self.stopped = False

    def should_stop(self, current_ppl: float) -> bool:
        if current_ppl < self.best_ppl:
            self.best_ppl = current_ppl
            self.counter = 0
        else:
            self.counter += 1

        self.stopped = self.counter >= self.patience
        return self.stopped


class Checkpoint:
    def __init__(self, args: Namespace):
        self.epoch = 0
        self.step = 0
        self.optimizer: Optional[dict] = None
        self.early_stopping = EarlyStopping(args.max_patience)
        self.train_stats = []
        self.losses = []
        self.unmasked_tokens = 0

class Checkpointer:
    def __init__(self, args: Namespace):
        self.checkpoint = Checkpoint(args)

    def update_checkpoint(
        self,
        model: PeftModel,
        optimizer: torch.optim.Optimizer,  # type: ignore
        step: int,
        losses: List[float],
        unmasked_tokens: int,
        output_folder: str,
        train_stats: Optional[List[dict]] = None,
        early_stopping: Optional[EarlyStopping] = None,
        epoch: Optional[int] = None,
    ):
        model.save_pretrained(os.path.join(output_folder, "checkpoint"))
        self.checkpoint.optimizer = optimizer.state_dict()
        if epoch is not None:
            self.checkpoint.epoch = epoch
        self.checkpoint.step = step
        self.checkpoint.losses = losses
        self.checkpoint.unmasked_tokens = unmasked_tokens
        if train_stats is not None:
            self.checkpoint.train_stats = train_stats
        if early_stopping is not None:
            self.checkpoint.early_stopping = early_stopping
        torch.save(
            self.checkpoint, os.path.join(output_folder, "checkpoint", "checkpoint.pt")
        )

    def load_checkpoint(
        self, model: AutoModelForCausalLM,
        output_folder: str,
        device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, PeftModel]:
        model = PeftModel.from_pretrained(
            model, os.path.join(output_folder, "checkpoint"), is_trainable=True  # type: ignore
        )
        if device is not None:
            self.checkpoint = torch.load(
                os.path.join(output_folder, "checkpoint", "checkpoint.pt"),
                weights_only=False,
                map_location=device
            )
        else:
            self.checkpoint = torch.load(
                os.path.join(output_folder, "checkpoint", "checkpoint.pt"),
                weights_only=False,
            )
        return self.checkpoint, model  # type: ignore


def resume_training(epoch: int, args: Namespace, early_stopping: EarlyStopping) -> bool:
    if epoch >= args.epochs:
        print(f"Reached maximum number of epochs {epoch}.")
        return False
    if early_stopping.stopped:
        print(f"Early stopping at epoch {epoch}")
        return False
    return True


def train_one_epoch(
    args,
    model: PeftModel,
    optimizer: torch.optim.Optimizer,  # type: ignore
    train_iterator: Iterator,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    start_step: int,
    steps_so_far: int,
    checkpointer: Checkpointer,
    output_folder: str,
    tokenizer: AutoTokenizer,
) -> Tuple[Tuple[float, float], int]:

    model.train()  # type: ignore
    losses = checkpointer.checkpoint.losses
    unmasked_tokens = checkpointer.checkpoint.unmasked_tokens

    for _ in range(start_step):
        next(train_iterator)

    with tqdm(dataloader, desc="Training") as pbar:
        pbar.total = len(dataloader)
        pbar.n = start_step
        pbar.refresh()
        for step, (input_ids, labels) in enumerate(train_iterator, start=start_step):
            if not isinstance(input_ids, dict):
                input_ids = input_ids.to(args.device)
            else:
                # molmo processor returns a dict
                input_ids = {k:v.to(args.device) for k,v in input_ids.items()}
            labels = labels.to(args.device)

            optimizer.zero_grad()

            outputs = model(**input_ids, labels=labels)

            loss = outputs.loss

            losses.append(loss.item())
            batch_unmasked_tokens = (labels != -100).sum().item()
            loss /= batch_unmasked_tokens
            loss.backward()
            optimizer.step()

            unmasked_tokens += batch_unmasked_tokens
            _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)
            pbar.set_postfix(
                {
                    "Train PPL": ppl,
                }
            )
            pbar.update(1)
            steps_so_far += 1

            if steps_so_far % args.save_every == 0:
                checkpointer.update_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    losses,
                    unmasked_tokens,
                    output_folder,
                )


    return compute_nll_and_ppl(losses, unmasked_tokens), steps_so_far


def train_one_epoch_clip(
    args,
    model: PeftModel,
    optimizer: torch.optim.Optimizer,  # type: ignore
    train_iterator: Iterator,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    start_step: int,
    steps_so_far: int,
    checkpointer: Checkpointer,
    output_folder: str,
    tokenizer: AutoTokenizer,
) -> Tuple[Tuple[float, float], int]:

    model.train()  # type: ignore
    losses = checkpointer.checkpoint.losses

    for _ in range(start_step):
        next(train_iterator)

    with tqdm(dataloader, desc="Training") as pbar:
        pbar.total = len(dataloader)
        pbar.n = start_step
        pbar.refresh()
        for step, (input_ids, text_candidates, target, target_id, sample_id) in enumerate(train_iterator, start=start_step):
            input_ids = input_ids.to(args.device)

            optimizer.zero_grad()

            output = model(**input_ids)

            assert target in tokenizer.decode(input_ids.input_ids[target_id])
            loss = criterion(output.logits_per_image, torch.tensor([target_id]).to(args.device))

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                {
                    "Train loss": mean(losses),
                }
            )
            pbar.update(1)
            steps_so_far += 1

            if steps_so_far % args.save_every == 0:
                checkpointer.update_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    losses,
                    -1, # no unmasked tokens
                    output_folder,
                )


    return mean(losses), steps_so_far



def train(
    args: Namespace,
    model: PeftModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,  # type: ignore
    output_folder: str,
    checkpointer: Checkpointer,
    tokenizer: AutoTokenizer,
    test: bool = False,
    clip: bool = False
) -> None:

    start_epoch = checkpointer.checkpoint.epoch
    start_step = checkpointer.checkpoint.step
    train_stats = checkpointer.checkpoint.train_stats
    early_stopping = checkpointer.checkpoint.early_stopping

    steps_so_far = len(train_loader) * start_epoch + start_step

    for _ in range(start_epoch):
        iter(train_loader)

    best_ppl = early_stopping.best_ppl
    best_loss = early_stopping.best_ppl
    for epoch in trange(
        start_epoch,
        args.epochs,
        desc="Epochs",
        initial=start_epoch,
        total=args.epochs,
    ):
        train_iterator = iter(train_loader)
        if not clip:
            (train_nnl, train_ppl), steps_so_far = train_one_epoch(
                args,
                model,
                optimizer,
                train_iterator,
                train_loader,
                criterion,
                start_step,
                steps_so_far,
                checkpointer,
                output_folder,
                tokenizer,
            )
            valid_nll, valid_ppl = evaluate(args, model, valid_loader, criterion, tokenizer=tokenizer)  # type: ignore

            if valid_ppl < best_ppl:
                best_ppl = valid_ppl
                model.save_pretrained(os.path.join(output_folder, "best_model"))

            should_stop = early_stopping.should_stop(valid_ppl)

            train_stats.append(
                {
                    "Epoch": epoch,
                    "Train NLL": train_nnl,
                    "Train PPL": train_ppl,
                    "Valid NLL": valid_nll,
                    "Valid PPL": valid_ppl,
                    "Patience": early_stopping.patience - early_stopping.counter,
                }
            )
        else:
            train_loss, steps_so_far = train_one_epoch_clip(
                args,
                model,
                optimizer,
                train_iterator,
                train_loader,
                criterion,
                start_step,
                steps_so_far,
                checkpointer,
                output_folder,
                tokenizer,
            )
            valid_loss = evaluate_clip(args, model, valid_loader, criterion, tokenizer=tokenizer)  # type: ignore

            if valid_loss < best_loss:
                best_ppl = valid_loss
                model.save_pretrained(os.path.join(output_folder, "best_model"))

            should_stop = early_stopping.should_stop(valid_loss)

            train_stats.append(
                {
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Valid Loss": valid_loss,
                    "Patience": early_stopping.patience - early_stopping.counter,
                }
            )

        checkpointer.update_checkpoint(
            model=model,
            optimizer=optimizer,
            step=0,
            losses=[],
            unmasked_tokens=0,
            output_folder=output_folder,
            train_stats=train_stats,
            early_stopping=early_stopping,
            epoch=epoch + 1,
        )

        start_step = 0

        with open(os.path.join(output_folder, "train_stats.json"), "w") as f:
            json.dump(train_stats, f, indent=4)

        if should_stop:
            break

        if test:
            break




def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
