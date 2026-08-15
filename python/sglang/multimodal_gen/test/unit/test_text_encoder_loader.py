import unittest
from types import SimpleNamespace
from unittest import mock

import transformers

from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
    _configure_text_encoder_quantization,
    _resolve_text_encoder_quant_config,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)


class TestTextEncoderClassResolution(unittest.TestCase):
    """load_native must not load encoder-decoder text encoders via AutoModel.

    AutoModel maps T5/UMT5 model types to the full seq2seq class
    (T5Model/UMT5Model), whose forward needs decoder inputs and raises when the
    module is used purely as a text encoder.
    """

    server_args = SimpleNamespace(trust_remote_code=False, revision=None)

    def _resolve(self, is_encoder_decoder, architectures):
        config = SimpleNamespace(
            is_encoder_decoder=is_encoder_decoder, architectures=architectures
        )
        with mock.patch.object(
            transformers.AutoConfig, "from_pretrained", return_value=config
        ):
            return TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )

    def test_umt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["UMT5EncoderModel"]), transformers.UMT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["UMT5Model"]), transformers.UMT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["UMT5ForConditionalGeneration"]),
            transformers.UMT5EncoderModel,
        )

    def test_t5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["T5EncoderModel"]), transformers.T5EncoderModel
        )
        self.assertIs(self._resolve(True, ["T5Model"]), transformers.T5EncoderModel)
        self.assertIs(
            self._resolve(True, ["T5ForConditionalGeneration"]),
            transformers.T5EncoderModel,
        )

    def test_mt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["MT5EncoderModel"]), transformers.MT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["MT5Model"]), transformers.MT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["MT5ForConditionalGeneration"]),
            transformers.MT5EncoderModel,
        )

    def test_non_encoder_decoder_keeps_automodel(self):
        # e.g. CLIP/Mistral/Qwen text encoders are not encoder-decoder.
        self.assertIs(self._resolve(False, ["CLIPTextModel"]), transformers.AutoModel)

    def test_unknown_architecture_falls_back_to_automodel(self):
        self.assertIs(self._resolve(True, ["NotARealClass"]), transformers.AutoModel)

    def test_config_load_failure_falls_back_to_automodel(self):
        with mock.patch.object(
            transformers.AutoConfig,
            "from_pretrained",
            side_effect=OSError("no config"),
        ):
            cls = TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )
        self.assertIs(cls, transformers.AutoModel)


class TestMiniMaxH3CheckpointFilter(unittest.TestCase):
    def test_only_known_unconsumed_weights_are_filtered(self):
        should_load = MiniMaxH3Qwen3VLEncoder.should_materialize_checkpoint_weight
        expected = {
            "model.language_model.layers.49.self_attn.q_proj.weight": True,
            "model.language_model.layers.50.self_attn.q_proj.weight": False,
            "model.language_model.layers.63.mlp.down_proj.weight": False,
            "model.language_model.norm.weight": False,
            "lm_head.weight": False,
            "model.language_model.rotary_emb.inv_freq": False,
            "model.visual.blocks.0.attn.qkv.weight": True,
            "language_model.layers.63.mlp.down_proj.weight": True,
            "module.model.language_model.layers.63.mlp.down_proj.weight": True,
        }
        self.assertEqual(
            {name: should_load(name) for name in expected},
            expected,
        )


class TestTextEncoderQuantization(unittest.TestCase):
    @staticmethod
    def _server_args(method=None, ignored_layers=None):
        return SimpleNamespace(
            text_encoder_quantization=method,
            text_encoder_quantization_ignored_layers=ignored_layers,
        )

    @mock.patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders."
        "text_encoder_loader.get_quant_config",
        return_value=None,
    )
    def test_online_fp8_uses_explicit_ignored_layers(self, _get_quant_config):
        ignored_layers = ["layers.0.self_attn"]
        quant_config = _resolve_text_encoder_quant_config(
            {},
            "/model/text_encoder",
            self._server_args("FP8", ignored_layers),
        )
        self.assertIsInstance(quant_config, Fp8Config)
        self.assertFalse(quant_config.is_checkpoint_fp8_serialized)
        self.assertEqual(quant_config.ignored_layers, ignored_layers)

    @mock.patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders."
        "text_encoder_loader.get_quant_config"
    )
    def test_serialized_checkpoint_config_takes_precedence(self, get_quant_config):
        serialized = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        get_quant_config.return_value = serialized
        resolved = _resolve_text_encoder_quant_config(
            {},
            "/model/text_encoder",
            self._server_args("fp8"),
        )
        self.assertIs(resolved, serialized)

    @mock.patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders."
        "text_encoder_loader.get_quant_config"
    )
    def test_serialized_checkpoint_rejects_online_ignored_layers(
        self, get_quant_config
    ):
        get_quant_config.return_value = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        )
        with self.assertRaisesRegex(ValueError, "only valid for online"):
            _resolve_text_encoder_quant_config(
                {},
                "/model/text_encoder",
                self._server_args("fp8", ["layers.0"]),
            )

    @mock.patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders."
        "text_encoder_loader.get_quant_config",
        return_value=None,
    )
    def test_encoder_class_must_opt_in(self, _get_quant_config):
        model_config = SimpleNamespace(quant_config=None)
        with self.assertRaisesRegex(ValueError, "does not support"):
            _configure_text_encoder_quantization(
                model_config,
                TextEncoder,
                {},
                "/model/text_encoder",
                self._server_args("fp8"),
            )

        _configure_text_encoder_quantization(
            model_config,
            MiniMaxH3Qwen3VLEncoder,
            {},
            "/model/text_encoder",
            self._server_args("fp8"),
        )
        self.assertIsInstance(model_config.quant_config, Fp8Config)


if __name__ == "__main__":
    unittest.main()
