from typing import Optional


class QwenVLReasoner:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        max_new_tokens: int = 512,
        torch_dtype: str = "auto",
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL requires torch, transformers, accelerate, and qwen-vl-utils. "
                "Install them with: pip install -r requirements.txt"
            ) from exc

        dtype = torch_dtype
        if torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype)

        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )

    def reason(self, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        })

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()
