import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set


def _find_mllm_raw_jsons(root: Path) -> List[Path]:
  """Recursively find all mllm_raw.json files under root."""
  if not root.exists() or not root.is_dir():
    return []
  return [p for p in root.rglob('mllm_raw.json') if p.is_file()]


def _safe_int(value: object) -> Optional[int]:
  try:
    return int(value) if value is not None else None
  except Exception:
    return None


def _load_sequence_usage(json_path: Path) -> Tuple[str, Dict[str, Optional[int]]]:
  """Load usage fields from a single mllm_raw.json file.

  Returns (scene_name, usage_dict) where usage_dict contains keys:
  input_tokens, input_text_tokens, input_visual_tokens (all Optional[int]).
  """
  try:
    with open(json_path, 'r') as f:
      data = json.load(f)
  except Exception:
    return (json_path.parent.name, {
      'input_tokens': None,
      'input_text_tokens': None,
      'input_visual_tokens': None,
    })

  scene_name = str(data.get('scene_name') or json_path.parent.name)
  usage = data.get('usage') or {}

  input_tokens = _safe_int(usage.get('input_tokens'))
  input_text_tokens = _safe_int(usage.get('input_text_tokens'))
  input_visual_tokens = _safe_int(usage.get('input_visual_tokens'))

  # If only one of text/visual is present and input_tokens is present, infer the missing one
  if input_tokens is not None:
    if input_text_tokens is None and input_visual_tokens is not None:
      input_text_tokens = max(0, input_tokens - input_visual_tokens)
    elif input_visual_tokens is None and input_text_tokens is not None:
      input_visual_tokens = max(0, input_tokens - input_text_tokens)

  return (scene_name, {
    'input_tokens': input_tokens,
    'input_text_tokens': input_text_tokens,
    'input_visual_tokens': input_visual_tokens,
  })


def _apply_fallbacks(stats: Dict[str, Optional[int]]) -> Dict[str, int]:
  """Apply fallback rules to ensure all three metrics are integers.

  Rule per user spec:
  - If BOTH input_text_tokens and input_visual_tokens are missing, set
    input_text_tokens = 1124 and input_visual_tokens = max(0, input_tokens - 1124).
  - If only one is missing and input_tokens is present, infer as difference.
  - If input_tokens is missing, treat it as 0 for computing visual fallback.
  """
  input_tokens = stats.get('input_tokens')
  text_tokens = stats.get('input_text_tokens')
  visual_tokens = stats.get('input_visual_tokens')

  if input_tokens is None:
    input_tokens = 0

  # Both missing → apply fixed text and estimate visual
  if text_tokens is None and visual_tokens is None:
    text_tokens = 1124
    visual_tokens = max(0, input_tokens - 1124)
  else:
    # If exactly one is missing, infer via difference where possible
    if text_tokens is None and visual_tokens is not None:
      text_tokens = max(0, input_tokens - visual_tokens)
    if visual_tokens is None and text_tokens is not None:
      visual_tokens = max(0, input_tokens - text_tokens)

  # Final guard: coerce to int, clamp negatives to zero
  input_tokens = int(input_tokens)
  text_tokens = int(text_tokens) if text_tokens is not None else 0
  visual_tokens = int(visual_tokens) if visual_tokens is not None else 0
  if input_tokens < 0:
    input_tokens = 0
  if text_tokens < 0:
    text_tokens = 0
  if visual_tokens < 0:
    visual_tokens = 0

  return {
    'input_tokens': input_tokens,
    'input_text_tokens': text_tokens,
    'input_visual_tokens': visual_tokens,
  }


def main():
  parser = argparse.ArgumentParser(description='Aggregate token usage across sequences produced by thinkvideo_mllm_keyframe_selector.py')
  parser.add_argument('--root_dir', type=str, required=True, help='Folder containing per-sequence subfolders with mllm_raw.json files')
  parser.add_argument('--output_path', type=str, default=None, help='Optional path to write aggregate_usage_summary.json (defaults to <root_dir>/aggregate_usage_summary.json)')
  parser.add_argument('--sequences_file', type=str, default=None, help='Optional text file listing scene names to include (one per line). Others are ignored.')
  args = parser.parse_args()

  root = Path(args.root_dir)
  json_paths = _find_mllm_raw_jsons(root)

  if not json_paths:
    print(f"No mllm_raw.json found under: {root}")
    sys.exit(1)

  include_set: Optional[Set[str]] = None
  if args.sequences_file:
    seq_file = Path(args.sequences_file)
    if not seq_file.exists():
      print(f"Sequences file not found: {seq_file}")
      sys.exit(1)
    try:
      with open(seq_file, 'r') as f:
        include_set = {line.strip() for line in f if line.strip() and not line.strip().startswith('#')}
    except Exception as e:
      print(f"Failed to read sequences file {seq_file}: {e}")
      sys.exit(1)

  per_sequence: List[Dict] = []
  sum_input = 0
  sum_text = 0
  sum_visual = 0

  for jp in sorted(json_paths):
    scene_name, raw_stats = _load_sequence_usage(jp)
    if include_set is not None and scene_name not in include_set:
      continue
    stats = _apply_fallbacks(raw_stats)
    per_sequence.append({
      'scene_name': scene_name,
      'json_path': str(jp),
      **stats,
    })
    sum_input += stats['input_tokens']
    sum_text += stats['input_text_tokens']
    sum_visual += stats['input_visual_tokens']
    print(f"[{scene_name}] input_tokens={stats['input_tokens']} input_text_tokens={stats['input_text_tokens']} input_visual_tokens={stats['input_visual_tokens']}")

  if not per_sequence:
    if include_set is not None:
      print(f"No sequences matched the include list ({len(include_set)} names) under: {root}")
    else:
      print("No sequences found to aggregate.")
    sys.exit(1)

  n = len(per_sequence)
  avg_input = round(sum_input / n, 4)
  avg_text = round(sum_text / n, 4)
  avg_visual = round(sum_visual / n, 4)

  print(f"Averages across {n} sequences → input_tokens={avg_input} input_text_tokens={avg_text} input_visual_tokens={avg_visual}")

  out_path = Path(args.output_path) if args.output_path else (root / 'aggregate_usage_summary.json')
  summary = {
    'root_dir': str(root),
    'num_sequences': n,
    'filter': {
      'sequences_file': str(args.sequences_file) if args.sequences_file else None,
      'num_include_names': len(include_set) if include_set is not None else None,
    },
    'averages': {
      'input_tokens': avg_input,
      'input_text_tokens': avg_text,
      'input_visual_tokens': avg_visual,
    },
    'sequences': per_sequence,
  }
  try:
    with open(out_path, 'w') as f:
      json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved aggregate summary → {out_path}")
  except Exception as e:
    print(f"Failed to write summary to {out_path}: {e}")


if __name__ == '__main__':
  main()


