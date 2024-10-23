from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    # Guiding text prompt
    model_output_path: Path = Path('evaluation','QBench')
    prompt_collection_path: Path = Path('prompt_collection','QBench')
    qa_model: str = 'allenai/unifiedqa-v2-t5-large-1363200'
    vqa_model: str = 'mplug-large'
    gpt_model: str= 'tifa-benchmark/llama2_tifa_question_generation'