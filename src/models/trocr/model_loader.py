# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Local Hugging Face TrOCR composition using editable DeiT and TrOCR modules."""

from __future__ import annotations

from transformers import VisionEncoderDecoderModel
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import (
    VisionEncoderDecoderConfig,
)

from .decoder.model import TrOCRForCausalLM
from .encoder.model import DeiTModel


class TrOCRVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    """Hugging Face-compatible TrOCR model with local architecture classes.

    Its state-dict key layout is identical to
    :class:`transformers.VisionEncoderDecoderModel`, so a Microsoft TrOCR
    checkpoint loads directly while ``encoder`` and ``decoder`` are instances
    of the editable local classes.
    """

    config_class = VisionEncoderDecoderConfig

    def __init__(
        self,
        config: VisionEncoderDecoderConfig,
        encoder: DeiTModel | None = None,
        decoder: TrOCRForCausalLM | None = None,
    ) -> None:
        if encoder is None:
            if config.encoder.model_type != "deit":
                raise ValueError(
                    "This local TrOCR implementation requires a DeiT encoder; "
                    f"received {config.encoder.model_type!r}."
                )
            encoder = DeiTModel(config.encoder)
        if decoder is None:
            if config.decoder.model_type != "trocr":
                raise ValueError(
                    "This local TrOCR implementation requires a TrOCR decoder; "
                    f"received {config.decoder.model_type!r}."
                )
            decoder = TrOCRForCausalLM(config.decoder)
        super().__init__(config=config, encoder=encoder, decoder=decoder)
