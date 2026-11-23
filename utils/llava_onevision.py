from qwen_vl_utils import process_vision_info

class GenerationCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        questions = []
        conversations = []
        texts = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            answers.append(answer)
            sample_ids.append(sample_id)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                    ],
                }
            ]

            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in conversations
        ]
        image_inputs, video_inputs = process_vision_info(conversations)
        input_ids = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        questions = self.processor.tokenizer(  # type: ignore
            questions, return_tensors="pt", padding=True
        )

        return input_ids, questions, answers, sample_ids


class TrainCollator:
    def __init__(
        self,
        processor,
        instruction="",
    ):
        self.processor = processor
        self.instruction = instruction

    def __call__(self, batch):
        questions = []
        conversations_no_answ = []
        conversations = []
        texts = []
        answers = []
        sample_ids = []

        for image, question, answer, sample_id in batch:
            questions.append(question)
            answers.append(answer)
            sample_ids.append(sample_id)

            conv_no_answ = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": (
                                f"{self.instruction} {question}"
                                if self.instruction
                                else question
                            ),
                        },
                    ],
                }
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

            conversations_no_answ.append(conv_no_answ)
            conversations.append(conversation)

        # add assistant prompt (will ignore it in training)
        texts_no_answ = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations_no_answ
        ]
        # -1 remove extra space/empty character at the end
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )[:-1]
            for conv in conversations
        ]


        image_inputs_no_answ, video_inputs_no_anw = process_vision_info(conversations_no_answ)
        input_no_answ = self.processor(
            text=texts_no_answ,
            images=image_inputs_no_answ,
            videos=video_inputs_no_anw,
            padding=True,
            return_tensors="pt",
        )

        image_inputs, video_inputs = process_vision_info(conversations)
        input_ids = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = input_ids.input_ids.detach().clone()

        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        no_answ_shape = input_no_answ.input_ids.shape
        labels[:no_answ_shape[0], :no_answ_shape[1]] = -100

        return input_ids, labels
