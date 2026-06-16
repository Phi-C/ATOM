"""SGLang external model package for ATOM plugin models."""


def register_minimax_m3_text_only_processor() -> None:
    """Let SGLang tokenizer init accept MiniMax-M3 text-only bring-up.

    MiniMax-M3 checkpoints advertise a conditional-generation architecture and
    include multimodal sub-configs, so SGLang asks for a multimodal processor
    before model workers are launched.  The ATOM SGLang path currently supports
    only the language model, so text-only serving just needs a processor object
    that will never be used for plain completion requests.
    """

    try:
        from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )
    except Exception:
        return

    class MiniMaxM3TextOnlyProcessor(BaseMultimodalProcessor):
        async def process_mm_data_async(self, *args, **kwargs):
            raise ValueError(
                "MiniMax-M3 SGLang ATOM currently supports text-only requests; "
                "multimodal inputs are not implemented."
            )

    class MiniMaxM3SparseForCausalLM:
        pass

    class MiniMaxM3SparseForConditionalGeneration:
        pass

    PROCESSOR_MAPPING.setdefault(MiniMaxM3SparseForCausalLM, MiniMaxM3TextOnlyProcessor)
    PROCESSOR_MAPPING.setdefault(
        MiniMaxM3SparseForConditionalGeneration,
        MiniMaxM3TextOnlyProcessor,
    )
