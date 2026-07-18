#!/usr/bin/env python3
"""Generate the exact pinned KVTuner GSM8K prompt fixture.

The script intentionally has no network or third-party dependency. Point it at the
`grade_school_math/data` directory from openai/grade-school-math commit
`b0bb162abedc65e1fdd8e93ed090fd7598ee68bc`; it rejects any different source bytes before
expanding the zero-shot sensitivity and seeded four-shot search prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


TRAIN_SHA256 = "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
TEST_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
SENSITIVITY_LIST_SHA256 = (
    "18d51be3aa1ac8e6ed7028a96c8c05efed1aa88588a2635e517d27a3e4e01730"
)
SEARCH_LIST_SHA256 = (
    "5e79ef00e8d8d602ce0b24a9ce49e2522fd5c775ae9e00f1e2c57f84931fb16e"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path, expected_sha256: str) -> list[dict[str, str]]:
    data = path.read_bytes()
    actual = sha256(data)
    if actual != expected_sha256:
        raise SystemExit(
            f"{path} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return [json.loads(line) for line in data.splitlines()]


def document_text(document: dict[str, str]) -> str:
    return f"Question: {document['question']}\nAnswer:"


def compact_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gsm8k-data",
        required=True,
        type=Path,
        help="Pinned grade_school_math/data directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    train = read_jsonl(args.gsm8k_data / "train.jsonl", TRAIN_SHA256)
    test = read_jsonl(args.gsm8k_data / "test.jsonl", TEST_SHA256)

    sensitivity = [document_text(document) for document in test[:20]]
    rng = random.Random(1234)
    search: list[str] = []
    few_shot_indices: list[list[int]] = []
    for document in test[:200]:
        indices = rng.sample(range(len(train)), 4)
        few_shot_indices.append(indices)
        context = "".join(
            document_text(train[index])
            + " "
            + train[index]["answer"]
            + "\n\n"
            for index in indices
        )
        search.append(context + document_text(document))

    if sha256(compact_json(sensitivity)) != SENSITIVITY_LIST_SHA256:
        raise SystemExit("expanded sensitivity prompt list does not match the pin")
    if sha256(compact_json(search)) != SEARCH_LIST_SHA256:
        raise SystemExit("expanded search prompt list does not match the pin")

    output = {
        "schemaVersion": 1,
        "fewShotSeed": 1234,
        "sensitivity": sensitivity,
        "search": search,
        "fewShotIndices": few_shot_indices,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compact_json(output) + b"\n")


if __name__ == "__main__":
    main()
