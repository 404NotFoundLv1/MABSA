"""VLP-MABSA-style readable output; structured records live in a separate file."""

import json
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    def __init__(self, output):
        self.output = Path(output)

    def info(self, message, pad=False):
        if pad:
            message = (' ' + message + ' ').center(50, '=')
        print(message, flush=True)
        with (self.output / 'log.txt').open('a', encoding='utf-8') as handle:
            handle.write(f'{datetime.now().isoformat(sep=" ", timespec="seconds")} INFO {message}\n')

    def event(self, record):
        with (self.output / 'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def joint_metrics(self, label, metrics):
        self.info(f"{label}  aesc_p:{metrics['precision'] * 100:.2f} "
                  f"aesc_r:{metrics['recall'] * 100:.2f} aesc_f:{metrics['f1'] * 100:.2f}")
