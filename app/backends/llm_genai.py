"""onnxruntime-genai LLM backend (Snapdragon X Elite NPU).

Validated against onnxruntime-genai 0.14.1 on this device (``og.is_qnn_available()``
returns True). The genai API used here is confirmed present in 0.14.x:

    Config:    og.Config(dir), .clear_providers(), .append_provider("QNNExecutionProvider")
    Model:     og.Model(config)   (falls back to og.Model(dir))
    Tokenizer: og.Tokenizer(model), .apply_chat_template(...), .encode(...), .create_stream()
    Generator: og.Generator(model, params); .append_tokens(ids);
               while not .is_done(): .generate_next_token(); .get_next_tokens()

The model directory (built via the Genie / genai export in WSL and copied to
``BRUTUS_LLM_MODEL_PATH``) must contain a ``genai_config.json``. Set
``BRUTUS_LLM_FORCE_PROVIDER=QNNExecutionProvider`` to pin generation to the NPU.

NOTE: this backend is fully wired but can only be exercised once the model bundle
is present on the device -- there is no bundle yet, so end-to-end generation is
unverified. Everything up to model load (QNN registration, is_qnn_available) is
confirmed on this hardware.
"""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, List

from app.backends.base import LLMBackend
from app.logging_conf import get_logger
from app.schemas import ChatMessage
from config import get_settings

log = get_logger("llm.genai")


class GenAILLM(LLMBackend):
    name = "genai"

    def __init__(self) -> None:
        self._og = None
        self._model = None
        self._tokenizer = None

    async def load(self) -> None:
        settings = get_settings()
        import onnxruntime_genai as og  # lazy: only imported in real mode

        self._og = og
        log.info("onnxruntime-genai %s, is_qnn_available=%s",
                 getattr(og, "__version__", "?"),
                 getattr(og, "is_qnn_available", lambda: "?")())

        # Register the QNN plugin EP with genai's runtime (best-effort; it may be
        # auto-loaded or already registered).
        if hasattr(og, "register_execution_provider_library"):
            try:
                import onnxruntime_qnn as oq

                og.register_execution_provider_library("QNNExecutionProvider", oq.get_library_path())
            except Exception as e:
                log.info("genai QNN EP registration skipped: %s", e)

        def _load():
            try:
                config = og.Config(settings.llm_model_path)
                if settings.llm_force_provider:
                    config.clear_providers()
                    config.append_provider(settings.llm_force_provider)
                    log.info("Forcing genai provider: %s", settings.llm_force_provider)
                model = og.Model(config)
            except Exception as e:
                # Fall back to letting genai_config.json drive provider selection.
                log.info("og.Config path failed (%s); loading model from dir directly", e)
                model = og.Model(settings.llm_model_path)
            tokenizer = og.Tokenizer(model)
            return model, tokenizer

        log.info("Loading genai model from %s ...", settings.llm_model_path)
        self._model, self._tokenizer = await asyncio.to_thread(_load)
        log.info("genai model loaded.")

    def _encode(self, messages: List[ChatMessage]):
        tokenizer = self._tokenizer
        # Prefer the model's own chat template for correct formatting.
        try:
            msg_json = json.dumps([{"role": m.role, "content": m.content} for m in messages])
            prompt = tokenizer.apply_chat_template(messages=msg_json, add_generation_prompt=True)
            if isinstance(prompt, str):
                return tokenizer.encode(prompt)
            return prompt  # some builds return token ids directly
        except Exception as e:
            log.info("apply_chat_template unavailable (%s); using generic template", e)
            parts = [f"<|{m.role}|>\n{m.content}" for m in messages]
            parts.append("<|assistant|>\n")
            return tokenizer.encode("\n".join(parts))

    async def stream(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[str]:
        og = self._og
        model = self._model
        tokenizer = self._tokenizer
        if og is None or model is None or tokenizer is None:
            raise RuntimeError("GenAILLM.load() was not called or failed")

        input_ids = self._encode(messages)

        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=len(input_ids) + max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

        generator = og.Generator(model, params)
        generator.append_tokens(input_ids)
        token_stream = tokenizer.create_stream()

        while not generator.is_done():
            # generate_next_token() is blocking C++; keep the event loop free.
            await asyncio.to_thread(generator.generate_next_token)
            new_token = generator.get_next_tokens()[0]
            piece = token_stream.decode(new_token)
            if piece:
                yield piece
