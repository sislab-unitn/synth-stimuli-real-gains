import re
from typing import List, Tuple

from PIL import Image
from transformers import CLIPProcessor


class SURECollator:
    def __init__(
        self,
        processor: CLIPProcessor,
        candidates: List[str],
    ):
        self.processor = processor
        self.candidates = candidates

    def __call__(self, batch: List[Tuple[Image.Image, str, str, str]]):

        assert len(batch) == 1, "SURECollator only supports batch size of 1"
        image, question, answer, sample_id = batch[0]

        # remove the candidates from the question
        question = re.sub(r" Choose from \[.*\].", "", question)
        assert "Choose from" not in question, f"candidates not removed from {question}"

        # create a candidate by combining the question and the candidate
        texts = [
            f"{question} {candidate.strip()}" for candidate in self.candidates
        ]

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
            input_data_format="channels_last",
        )

        answer_id = self.candidates.index(answer)

        return inputs, texts, answer, answer_id, sample_id
