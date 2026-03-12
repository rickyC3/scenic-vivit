"""ViViT 批量推理和準確率統計（支持自定義 label2idx 映射）。"""

import csv
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from absl import app, flags
import cv2
import ffmpeg
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import qwix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to ViViT checkpoint")
flags.DEFINE_string("video_dir", None, "Directory containing videos")
flags.DEFINE_string("csv_path", None, "Path to CSV with labels (video_path, start, end, label)")
flags.DEFINE_string("label_map_path", None, "Path to label2idx mapping file (text format)")
flags.DEFINE_string("output_dir", "./vivit_results", "Output directory for results")
flags.DEFINE_integer("num_frames", 32, "Number of frames to extract")
flags.DEFINE_integer("frame_size", 224, "Frame size (height/width)")
flags.DEFINE_integer("num_workers", 4, "Number of parallel workers")
flags.DEFINE_boolean("save_predictions", True, "Save predictions to CSV")
flags.DEFINE_enum(
    "extract_backend",
    "cv2",
    ["cv2", "ffmpeg"],
    "Frame extraction backend. 'cv2' uses OpenCV seeks; 'ffmpeg' uses ffmpeg filter pipeline.",
)


class LabelManager:
  """管理標籤映射。"""

  def __init__(self, label_map_path: Optional[str] = None):
    self.label_to_idx = {}
    self.idx_to_label = {}

    if label_map_path and Path(label_map_path).exists():
      self._load_from_file(label_map_path)
    else:
      logger.warning("⚠️  未提供標籤映射文件")

  def _load_from_file(self, label_map_path: str):
    try:
      file_ext = Path(label_map_path).suffix.lower()

      if file_ext == '.txt':
        self._load_from_txt(label_map_path)
      elif file_ext == '.json':
        self._load_from_json(label_map_path)
      elif file_ext == '.csv':
        self._load_from_csv(label_map_path)
      else:
        logger.warning(f"⚠️  未知的文件格式 {file_ext}，嘗試按 TXT 格式讀取")
        self._load_from_txt(label_map_path)

      logger.info(f"✅ 成功加載 {len(self.label_to_idx)} 個標籤映射")

    except Exception as e:
      logger.error(f"❌ 加載標籤映射失敗: {e}")

  def _load_from_txt(self, label_map_path: str):
    with open(label_map_path, 'r', encoding='utf-8') as f:
      # TODO: 確認順序是否正確，或者需要根據文件內容解析出 idx 和 label
      for idx, line in enumerate(f):  
        label_name = line.strip()
        if label_name:
          self.label_to_idx[label_name] = idx
          self.idx_to_label[idx] = label_name

  def _load_from_json(self, label_map_path: str):
    with open(label_map_path, 'r', encoding='utf-8') as f:
      data = json.load(f)

    if data:
      first_value = next(iter(data.values()))
      if isinstance(first_value, int):
        self.label_to_idx = data
        self.idx_to_label = {v: k for k, v in data.items()}
      else:
        self.idx_to_label = {int(k): v for k, v in data.items()}
        self.label_to_idx = {v: int(k) for k, v in data.items()}

  def _load_from_csv(self, label_map_path: str):
    with open(label_map_path, 'r', encoding='utf-8') as f:
      reader = csv.DictReader(f)
      if reader.fieldnames:
        if 'id' in reader.fieldnames and 'name' in reader.fieldnames:
          id_col, name_col = 'id', 'name'
        elif 'index' in reader.fieldnames and 'label' in reader.fieldnames:
          id_col, name_col = 'index', 'label'
        else:
          id_col, name_col = reader.fieldnames[0], reader.fieldnames[1]

        for row in reader:
          idx = int(row[id_col])
          name = row[name_col]
          self.label_to_idx[name] = idx
          self.idx_to_label[idx] = name

  def get_label(self, idx: int) -> Optional[str]:
    return self.idx_to_label.get(idx)

  def num_classes(self) -> int:
    return len(self.label_to_idx)

  def print_sample(self, num_samples: int = 10):
    logger.info(f"\n📋 標籤映射樣本 (前 {num_samples} 個):")
    for idx in range(min(num_samples, len(self.idx_to_label))):
      label = self.idx_to_label.get(idx, "N/A")
      logger.info(f"   {idx}: {label}")


def extract_frames_cv2(video_path: str,
                       num_frames: int = 32,
                       frame_size: int = 224,
                       start_time: float = 0.0,
                       end_time: Optional[float] = None) -> Optional[np.ndarray]:
  """使用 OpenCV 提取幀。"""
  try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 10.0

    start_time = max(0.0, start_time)
    if end_time is None:
      end_time = duration
    else:
      end_time = min(duration, end_time)

    if start_time >= end_time:
      cap.release()
      return None

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_indices = np.linspace(start_frame, max(start_frame, end_frame - 1), num_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
      cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
      ret, frame = cap.read()
      if not ret:
        frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
      else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
      frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0).astype(np.float32)
  except Exception as e:
    logger.debug(f"提取幀失敗(cv2): {e}")
    return None


def extract_frames_ffmpeg(video_path: str,
                          num_frames: int = 32,
                          frame_size: int = 224,
                          start_time: float = 0.0,
                          end_time: Optional[float] = None) -> Optional[np.ndarray]:
  """使用 ffmpeg 提取固定數量 RGB 幀。"""
  try:
    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe.get('streams', []) if s.get('codec_type') == 'video'), None)
    if not video_stream:
      return None

    duration = float(video_stream.get('duration', probe.get('format', {}).get('duration', 0.0) or 0.0))
    if duration <= 0:
      return None

    start_time = max(0.0, start_time)
    end_time = min(duration, end_time if end_time is not None else duration)
    if start_time >= end_time:
      return None

    clip_duration = end_time - start_time
    target_fps = max(1e-6, num_frames / clip_duration)

    out, _ = (
        ffmpeg
        .input(video_path, ss=start_time, t=clip_duration)
        .filter('fps', fps=target_fps)
        .filter('scale', frame_size, frame_size)
        .output('pipe:', format='yuvj420p', pix_fmt='rgb24', vframes=num_frames)
        .run(capture_stdout=True, capture_stderr=True, quiet=True)
    )

    frame_bytes = frame_size * frame_size * 3
    if not out:
      return None

    total_frames = len(out) // frame_bytes
    if total_frames <= 0:
      return None

    frames = np.frombuffer(out, np.uint8).reshape((total_frames, frame_size, frame_size, 3))

    if total_frames < num_frames:
      pad = np.repeat(frames[-1][np.newaxis, ...], num_frames - total_frames, axis=0)
      frames = np.concatenate([frames, pad], axis=0)
    elif total_frames > num_frames:
      idx = np.linspace(0, total_frames - 1, num_frames, dtype=int)
      frames = frames[idx]
    logging.info(f"提取幀成功(ffmpeg) at line 224: {frames.shape}")
    return frames.astype(np.float32)
      
  except ffmpeg.Error as e:
    logging.info(f"提取幀失敗(ffmpeg) at line 228: {e.stderr.decode()}")
    logger.debug(f"提取幀失敗(ffmpeg): {e}")
    return None
  except Exception as e:
    logger.debug(f"提取幀失敗(ffmpeg): {e}")
    return None


def extract_frames(video_path: str,
                   num_frames: int = 32,
                   frame_size: int = 224,
                   start_time: float = 0.0,
                   end_time: Optional[float] = None,
                   backend: str = 'cv2') -> Optional[np.ndarray]:
  """提取幀（可切換 cv2 / ffmpeg）。"""
  if backend == 'ffmpeg':
    return extract_frames_ffmpeg(video_path, num_frames, frame_size, start_time, end_time)
  return extract_frames_cv2(video_path, num_frames, frame_size, start_time, end_time)


def normalize_frames(frames: np.ndarray) -> np.ndarray:
  frames = frames / 255.0
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
  std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
  return (frames - mean) / std


def prepare_batch(frames: np.ndarray, normalize: bool = True) -> jnp.ndarray:
  if normalize:
    frames = normalize_frames(frames)
  return jnp.expand_dims(frames, axis=0)


def load_vivit_model(checkpoint_path: str, config: ml_collections.ConfigDict):
  del config
  try:
    checkpoint_data = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    if checkpoint_data is None:
      return None, None
    logging.info(checkpoint_data.keys())
    if ('params' in checkpoint_data) and ('model_state' in checkpoint_data):
      params = checkpoint_data['params']
      model_state = checkpoint_data['model_state']
    else:
      params = checkpoint_data['optimizer']["target"]
      model_state = checkpoint_data['model_state']
    return params, model_state
  except Exception as e:
    logger.error(f"加載模型失敗: {e}")
    return None, None


def vivit_forward_pass(batch: jnp.ndarray, params, model_state, flax_model,
                       deterministic: bool = True) -> Optional[jnp.ndarray]:
  try:
    variables = {'params': params, **model_state}
    return flax_model.apply(variables, batch, train=False)
  except Exception as e:
    logger.error(f"前向傳播失敗: {e}")
    return None


class VideoInferenceResult:
  def __init__(self, video_path: str, video_name: str, label: Optional[str] = None):
    self.video_path = video_path
    self.video_name = video_name
    self.label = label
    self.predicted_idx = None
    self.predicted_label = None
    self.confidence = None
    self.top5_preds = []
    self.success = False
    self.error_msg = None

  def to_dict(self) -> Dict:
    top5_str = ", ".join([f"{l}({p:.4f})" for l, p in self.top5_preds])
    return {
        'video_name': self.video_name,
        'true_label': self.label or 'N/A',
        'predicted_label': self.predicted_label or 'N/A',
        'confidence': f"{self.confidence:.4f}" if self.confidence else 'N/A',
        'top5_predictions': top5_str or 'N/A',
        'correct': str(self.label == self.predicted_label) if self.label else 'N/A',
        'error': self.error_msg or 'None'
    }


def infer_single_video(video_info: Tuple, params, model_state, flax_model,
                       label_manager: LabelManager, num_frames: int = 32,
                       frame_size: int = 224, extract_backend: str = 'cv2') -> VideoInferenceResult:
  video_path, start, end, label = video_info
  result = VideoInferenceResult(video_path, Path(video_path).name, label)

  try:
    frames = extract_frames(
        video_path,
        num_frames=num_frames,
        frame_size=frame_size,
        start_time=float(start),
        end_time=float(end) if end is not None else None,
        backend=extract_backend,
    )

    if frames is None:
      result.error_msg = f"幀提取失敗({extract_backend})"
      return result

    #logging.info("checkpoint 2: successfully extracted frames with %s", extract_backend)

    batch = prepare_batch(frames, normalize=True)
    logits = vivit_forward_pass(batch, params, model_state, flax_model)
    if logits is None:
      result.error_msg = "前向傳播失敗"
      return result

    #logging.info("checkpoint 3: successfully got logits from model")

    probs = jax.nn.softmax(logits[0])
    pred_idx = int(jnp.argmax(probs))
    result.predicted_idx = pred_idx
    result.predicted_label = label_manager.get_label(pred_idx)
    result.confidence = float(probs[pred_idx])

    #logging.info("checkpoint 4: predicted label = %s with confidence %.4f", result.predicted_label, result.confidence)

    top5_indices = jnp.argsort(probs)[-5:][::-1]
    for idx in top5_indices:
      idx = int(idx)
      result.top5_preds.append((label_manager.get_label(idx), float(probs[idx])))

    logging.info("checkpoint 5: top-5 predictions = %s", result.top5_preds)

    result.success = True
  except Exception as e:
    result.error_msg = str(e)
    logger.error(f"推理 {video_path} 失敗: {e}")
  return result


def batch_inference(video_infos: List[Tuple], checkpoint_path: str,
                    config: ml_collections.ConfigDict,
                    label_manager: LabelManager,
                    num_workers: int = 4,
                    num_frames: int = 32,
                    frame_size: int = 224,
                    extract_backend: str = 'cv2') -> List[VideoInferenceResult]:
  params, model_state = load_vivit_model(checkpoint_path, config)
  if params is None:
    logger.error("❌ 無法加載模型")
    return []
  else:
    logger.info("✅ Get params and model state from checkpoint successfully.")
  from scenic.projects.vivit import model as vivit_model_lib
  flax_model = vivit_model_lib.get_model_cls(config.model_name)(
      config, {'num_classes': config.num_classes}).flax_model

  results = []
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = {
        executor.submit(
            infer_single_video,
            video_info,
            params,
            model_state,
            flax_model,
            label_manager,
            num_frames,
            frame_size,
            extract_backend,
        ): i
        for i, video_info in enumerate(video_infos)
    }

    with tqdm(total=len(video_infos), desc="推理進度") as pbar:
      for future in as_completed(futures):
        results.append(future.result())
        pbar.update(1)
  return results


class AccuracyStatistics:
  def __init__(self, results: List[VideoInferenceResult]):
    self.results = results
    self.calculate()

  def calculate(self):
    self.total = len(self.results)
    self.successful = sum(1 for r in self.results if r.success)
    self.failed = self.total - self.successful
    valid_results = [r for r in self.results if r.success and r.label]
    self.evaluated = len(valid_results)

    if self.evaluated > 0:
      self.correct_count = sum(1 for r in valid_results if r.label == r.predicted_label)
      self.accuracy = self.correct_count / self.evaluated
      self.top5_correct_count = sum(
          1 for r in valid_results if any(label == r.label for label, _ in r.top5_preds))
      self.top5_accuracy = self.top5_correct_count / self.evaluated
    else:
      self.correct_count = 0
      self.accuracy = 0.0
      self.top5_correct_count = 0
      self.top5_accuracy = 0.0

    self.per_class_stats = {}
    for result in valid_results:
      self.per_class_stats.setdefault(result.label, {'total': 0, 'correct': 0})
      self.per_class_stats[result.label]['total'] += 1
      if result.label == result.predicted_label:
        self.per_class_stats[result.label]['correct'] += 1

  def print_summary(self):
    logger.info("\n" + "=" * 60)
    logger.info("📊 推理統計結果")
    logger.info("=" * 60)
    logger.info(f"\n總視頻數: {self.total}")
    logger.info(f"推理成功: {self.successful}")
    logger.info(f"推理失敗: {self.failed}")
    logger.info(f"評估樣本: {self.evaluated}")
    if self.evaluated > 0:
      logger.info(f"\n🎯 Top-1 準確率: {self.accuracy * 100:.2f}% ({self.correct_count}/{self.evaluated})")
      logger.info(f"🎯 Top-5 準確率: {self.top5_accuracy * 100:.2f}% ({self.top5_correct_count}/{self.evaluated})")
    logger.info("=" * 60 + "\n")


def load_csv_labels(csv_path: str, video_dir: str) -> List[Tuple]:
  try:
    df = pd.read_csv(csv_path)
    video_infos = []
    for _, row in df.iterrows():
      video_full_path = os.path.join(video_dir, row['video_path'])
      if not Path(video_full_path).exists():
        logger.warning(f"⚠️  視頻不存在: {video_full_path}")
        continue
      video_infos.append((video_full_path, float(row['start']), float(row['end']), row.get('label', None)))
    logger.info(f"✅ 加載 {len(video_infos)} 個視頻標籤")
    return video_infos
  except Exception as e:
    logger.error(f"加載 CSV 失敗: {e}")
    return []


def get_all_videos_in_dir(video_dir: str) -> List[Tuple]:
  video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
  video_infos = []
  for filename in sorted(os.listdir(video_dir)):
    if Path(filename).suffix.lower() in video_extensions:
      video_infos.append((os.path.join(video_dir, filename), 0.0, None, None))
  return video_infos


def save_results_to_csv(results: List[VideoInferenceResult], output_path: str):
  pd.DataFrame([r.to_dict() for r in results]).to_csv(output_path, index=False, encoding='utf-8')
  logger.info(f"✅ 結果已保存到: {output_path}")


def main(argv):
  del argv
  if not FLAGS.checkpoint_path:
    raise ValueError("必須指定 --checkpoint_path")
  if not FLAGS.video_dir:
    raise ValueError("必須指定 --video_dir")
  if not FLAGS.label_map_path:
    raise ValueError("必須指定 --label_map_path")

  output_dir = Path(FLAGS.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  label_manager = LabelManager(FLAGS.label_map_path)
  label_manager.print_sample(num_samples=10)

  if FLAGS.csv_path and Path(FLAGS.csv_path).exists():
    video_infos = load_csv_labels(FLAGS.csv_path, FLAGS.video_dir)
  else:
    video_infos = get_all_videos_in_dir(FLAGS.video_dir)

  if not video_infos:
    logger.error("❌ 沒有找到視頻")
    return

  config = ml_collections.ConfigDict({
      'model_name': 'vivit_classification',
      'num_classes': 400, #label_manager.num_classes(),
      'model': ml_collections.ConfigDict()
  })
  config.model.hidden_size = 768
  config.model.attention_config = ml_collections.ConfigDict()
  config.model.attention_config.type = 'factorized_encoder'
  config.model.patches = ml_collections.ConfigDict()
  config.model.spatial_transformer = ml_collections.ConfigDict()
  config.model.spatial_transformer.num_heads = 12
  config.model.spatial_transformer.mlp_dim = 3072
  config.model.spatial_transformer.num_layers = 12
  config.model.temporal_transformer = ml_collections.ConfigDict()
  config.model.temporal_transformer.num_heads = 12
  config.model.temporal_transformer.mlp_dim = 3072
  config.model.temporal_transformer.num_layers = 12
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.1
  config.model.dropout_rate = 0.1
  config.model_dtype_str = 'float32'
  config.model.temporal_encoding_config = ml_collections.ConfigDict()
  config.model.temporal_encoding_config.method = '3d_conv'
  config.model.patches.size = (16, 16, 2)
  config.model.temporal_encoding_config.kernel_init_method = 'central_frame_initializer'
  # Applies when temporal_encoding_config.method='temporal_sampling'
  config.model.temporal_encoding_config.n_sampled_frames = 16  # Unused here.

  results = batch_inference(
      video_infos,
      FLAGS.checkpoint_path,
      config,
      label_manager,
      num_workers=FLAGS.num_workers,
      num_frames=FLAGS.num_frames,
      frame_size=FLAGS.frame_size,
      extract_backend=FLAGS.extract_backend,
  )

  stats = AccuracyStatistics(results)
  stats.print_summary()

  if FLAGS.save_predictions:
    save_results_to_csv(results, str(output_dir / "predictions_0304.csv"))

  logger.info(f"✅ 推理完成！結果保存在: {output_dir}")


if __name__ == "__main__":
  app.run(main)

"""
python run_inference_batch_v3.py \
  --checkpoint_path=/users/undergraduate/rjchen25/scenic-vivit/train/train_0301_5/checkpoint_114001 \
  --video_dir=/users/undergraduate/rjchen25/kinetic400-test/test_datasets/videos_val/ \
  --csv_path=/users/undergraduate/rjchen25/kinetic400-test/test_datasets/output.csv \
  --label_map_path=/users/undergraduate/rjchen25/k400/csv_files/kinetics_400_labels.csv \
  --output_dir=./vivit_results \
  --num_frames=32 \
  --frame_size=224 \
  --num_workers=4 \
  --extract_backend=ffmpeg > inf_0304.log 2>&1
"""