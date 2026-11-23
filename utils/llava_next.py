from transformers import LlavaNextProcessor

class GenerationCollator:
    def __init__(
        self,
        processor: LlavaNextProcessor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):

        samples = []
        questions = []
        images = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            images.append(image)
            answers.append(answer)
            sample_ids.append(sample_id)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                        {"type": "image"},
                    ],
                },
            ]

            sample = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True, # used in the huggingface example
            )
            samples.append(sample)

        input_ids = self.processor(
            text=samples, images=images, return_tensors="pt", padding=True
        )
        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids


class TrainCollator:
    def __init__(
        self,
        processor: LlavaNextProcessor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):

        samples = []
        samples_no_answ = []
        questions = []
        images = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            images.append(image)
            answers.append(answer)
            sample_ids.append(sample_id)

            conv_no_answ = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                        {"type": "image"},
                    ],
                },
            ]
            conversation = [
                conv_no_answ[0],
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer
                        }
                    ]
                }
            ]

            sample_no_answ = self.processor.apply_chat_template(
                conv_no_answ,
                add_generation_prompt=True,
            )
            samples_no_answ.append(sample_no_answ)

            # -1 remove extra space/empty character at the end
            sample = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
            )[:-1]
            # add eos_token to the sample string (learn to end generation)
            sample = sample + self.processor.tokenizer.eos_token
            samples.append(sample)

        input_no_answ = []
        for s_no_ans, img in zip(samples_no_answ, images):
            input_no_answ.append(self.processor(text=s_no_ans, images=img, return_tensors="pt", padding=True))
        input_ids = self.processor(text=samples, images=images, return_tensors="pt", padding=True)

        labels = input_ids.input_ids.detach().clone()

        # ignore pad
        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        # ignore image and question
        for i, in_no_answ in enumerate(input_no_answ):
            no_answ_shape = in_no_answ.input_ids.shape
            labels[i, :no_answ_shape[1]] = -100

        return input_ids, labels
