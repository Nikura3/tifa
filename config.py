from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from enum import Enum

class TifaVersion(Enum): REGULAR = "REGULAR"; EXTENDED = "EXTENDED"


@dataclass
class RunConfig:
    # Guiding text prompt
    eval_path: Path = Path('evaluation','QBench')
    qa_model: str = 'allenai/unifiedqa-v2-t5-large-1363200'
    vqa_model: str = 'mplug-large'
    gpt_model: str= 'tifa-benchmark/llama2_tifa_question_generation'
    tifa_version: TifaVersion = TifaVersion.EXTENDED
