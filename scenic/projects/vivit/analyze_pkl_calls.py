#!/usr/bin/env python3
"""Analyze `__call__` distribution following specific ViViT layer markers in a pickle file.

This script scans a pickle object recursively and builds an ordered token stream from:
- dict keys
- string-like leaf values

For each occurrence of the markers below, it finds the *next* token that contains
`__call__` and aggregates per-layer statistics / distributions:
- layer_norm_input_1
- MHA_input
- layer_norm_input_2
- MLP_input
"""

from __future__ import annotations

import argparse
import collections
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TARGET_MARKERS: Tuple[str, ...] = (
    "layer_norm_input_1",
    "MHA_input",
    "layer_norm_input_2",
    "MLP_input",
)
CALL_PATTERN = "__call__"


def _iter_tokens(obj: Any, path: str = "root") -> Iterable[Tuple[str, str]]:
  """Recursively yields (path, token) pairs in traversal order."""
  if isinstance(obj, dict):
    for k, v in obj.items():
      key_text = str(k)
      yield f"{path}.<key>", key_text
      yield from _iter_tokens(v, f"{path}[{key_text!r}]")
  elif isinstance(obj, (list, tuple)):
    for i, v in enumerate(obj):
      yield from _iter_tokens(v, f"{path}[{i}]")
  elif isinstance(obj, set):
    # Set iteration order is not guaranteed, but we still support it as best effort.
    for i, v in enumerate(obj):
      yield from _iter_tokens(v, f"{path}<set:{i}>")
  elif isinstance(obj, bytes):
    try:
      yield path, obj.decode("utf-8", errors="replace")
    except Exception:
      yield path, str(obj)
  elif isinstance(obj, str):
    yield path, obj
  else:
    # Ignore non-string leaves to avoid noisy tokenization.
    return


def _extract_call_name(token: str) -> str:
  """Extracts a compact `...__call__...` segment for distribution aggregation."""
  compact = token.strip()
  match = re.search(r"([\w./:-]*__call__[\w./:-]*)", compact)
  if match:
    return match.group(1)
  return compact


def _extract_encoder_id(path: str, token: str) -> str:
  """Best-effort encoder index extraction from path/token text.

  Common patterns we try to capture:
  - encoderblock_0
  - encoder_block_0
  - encoder/0
  - layer_0
  """
  text = f"{path} {token}"
  patterns = (
      r"encoderblock[_:/-]?(\d+)",
      r"encoder[_-]?block[_:/-]?(\d+)",
      r"encoder[_:/-](\d+)",
      r"layer[_:/-]?(\d+)",
  )
  for p in patterns:
    m = re.search(p, text, flags=re.IGNORECASE)
    if m:
      return m.group(1)
  return "unknown"


def analyze_markers_to_next_call(
    tokens: Sequence[Tuple[str, str]],
    markers: Sequence[str],
    call_pattern: str = CALL_PATTERN,
) -> Dict[str, Any]:
  """Finds next call token after each marker and computes distributions."""
  per_marker_call_hits: Dict[str, collections.Counter[str]] = {
      m: collections.Counter() for m in markers
  }
  per_marker_examples: Dict[str, List[Dict[str, str]]] = {m: [] for m in markers}
  per_encoder_marker_call_hits: Dict[str, Dict[str, collections.Counter[str]]] = (
      collections.defaultdict(lambda: {m: collections.Counter() for m in markers})
  )
  per_encoder_marker_count: Dict[str, collections.Counter[str]] = (
      collections.defaultdict(collections.Counter)
  )
  per_encoder_marker_missing: Dict[str, collections.Counter[str]] = (
      collections.defaultdict(collections.Counter)
  )
  marker_count = collections.Counter()
  marker_no_following_call = collections.Counter()

  for i, (marker_path, marker_token) in enumerate(tokens):
    hit_markers = [m for m in markers if m in marker_token]
    if not hit_markers:
      continue

    for marker in hit_markers:
      encoder_id = _extract_encoder_id(marker_path, marker_token)
      marker_count[marker] += 1
      per_encoder_marker_count[encoder_id][marker] += 1
      found: Optional[Tuple[str, str]] = None
      for j in range(i + 1, len(tokens)):
        call_path, call_token = tokens[j]
        if call_pattern in call_token:
          found = (call_path, call_token)
          break

      if found is None:
        marker_no_following_call[marker] += 1
        per_encoder_marker_missing[encoder_id][marker] += 1
        continue

      call_path, call_token = found
      call_name = _extract_call_name(call_token)
      per_marker_call_hits[marker][call_name] += 1
      per_encoder_marker_call_hits[encoder_id][marker][call_name] += 1
      if len(per_marker_examples[marker]) < 5:
        per_marker_examples[marker].append({
            "marker_path": marker_path,
            "marker_token": marker_token,
            "call_path": call_path,
            "encoder_id": encoder_id,
            "call_token": call_token,
        })

  result = {}
  for marker in markers:
    total = marker_count[marker]
    call_hits = per_marker_call_hits[marker]
    matched = sum(call_hits.values())
    distribution = []
    for name, cnt in call_hits.most_common():
      distribution.append({
          "call_name": name,
          "count": cnt,
          "ratio": (cnt / matched) if matched else 0.0,
      })

    result[marker] = {
        "marker_occurrences": total,
        "matched_next_call": matched,
        "missing_next_call": marker_no_following_call[marker],
        "distribution": distribution,
        "examples": per_marker_examples[marker],
    }
  per_encoder = {}
  for encoder_id in sorted(per_encoder_marker_count.keys(), key=str):
    per_encoder[encoder_id] = {}
    for marker in markers:
      call_hits = per_encoder_marker_call_hits[encoder_id][marker]
      matched = sum(call_hits.values())
      distribution = [
          {
              "call_name": name,
              "count": cnt,
              "ratio": (cnt / matched) if matched else 0.0,
          }
          for name, cnt in call_hits.most_common()
      ]
      per_encoder[encoder_id][marker] = {
          "marker_occurrences": per_encoder_marker_count[encoder_id][marker],
          "matched_next_call": matched,
          "missing_next_call": per_encoder_marker_missing[encoder_id][marker],
          "distribution": distribution,
      }
  result["per_encoder"] = per_encoder
  return result


def _print_report(report: Dict[str, Any]) -> None:
  print("=== Next '__call__' distribution report ===")
  for marker in TARGET_MARKERS:
    stats = report[marker]
    print(f"\n[{marker}]")
    print(
        "  occurrences={occ}, matched={matched}, missing={missing}".format(
            occ=stats["marker_occurrences"],
            matched=stats["matched_next_call"],
            missing=stats["missing_next_call"],
        ))

    if not stats["distribution"]:
      print("  distribution: <empty>")
      continue

    print("  distribution:")
    for item in stats["distribution"]:
      print(
          "    - {name}: count={count}, ratio={ratio:.4f}".format(
              name=item["call_name"], count=item["count"], ratio=item["ratio"]
          ))

  per_encoder = report.get("per_encoder", {})
  if per_encoder:
    print("\n=== Per-encoder breakdown ===")
    for encoder_id in sorted(per_encoder.keys(), key=str):
      print(f"\n[encoder={encoder_id}]")
      for marker in TARGET_MARKERS:
        stats = per_encoder[encoder_id][marker]
        print(
            "  {marker}: occurrences={occ}, matched={matched}, missing={missing}".
            format(
                marker=marker,
                occ=stats["marker_occurrences"],
                matched=stats["matched_next_call"],
                missing=stats["missing_next_call"],
            ))
        if stats["distribution"]:
          for item in stats["distribution"]:
            print(
                "    - {name}: count={count}, ratio={ratio:.4f}".format(
                    name=item["call_name"],
                    count=item["count"],
                    ratio=item["ratio"],
                ))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Read a pickle file and find the first '__call__' token that appears "
          "after target markers."
      )
  )
  parser.add_argument("pkl_path", type=Path, help="Path to .pkl file")
  parser.add_argument(
      "--save-report",
      type=Path,
      default=None,
      help="Optional path to save the parsed report as .pkl",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  with args.pkl_path.open("rb") as f:
    data = pickle.load(f)

  tokens = list(_iter_tokens(data))
  report = analyze_markers_to_next_call(tokens, TARGET_MARKERS)
  _print_report(report)

  if args.save_report is not None:
    with args.save_report.open("wb") as f:
      pickle.dump(report, f)
    print(f"\nSaved report to: {args.save_report}")


if __name__ == "__main__":
  main()
