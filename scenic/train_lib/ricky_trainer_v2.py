# Copyright 2025 The Scenic Authors.
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimized training script (v2) with safer defaults for single-GPU runs."""

import copy
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
import jax
from jax.example_libraries.optimizers import clip_grads
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
import optax

from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]


def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: LossFn,
    lr_fn: LrFn,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
    use_pmap: bool = True,
) -> Tuple[train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str, Any]]:
  """Runs a single step of training."""
  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

  if config.get('mixup') and config.mixup.alpha:
    mixup_rng, rng = jax.random.split(rng, 2)
    if use_pmap:
      mixup_rng = train_utils.bind_rng_to_host_device(
          mixup_rng,
          axis_name='batch',
          bind_to=config.mixup.get('bind_to', 'device'))
    batch = dataset_utils.mixup(
        batch,
        config.mixup.alpha,
        config.mixup.get('image_format', 'NTHWC'),
        rng=mixup_rng)

  dropout_rng = (
      train_utils.bind_rng_to_host_device(rng, axis_name='batch', bind_to='device')
      if use_pmap else rng
  )

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}
    logits, new_model_state = flax_model.apply(
        variables,
        batch['inputs'],
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug)
    loss = loss_fn(logits, batch, variables['params'])
    return loss, (new_model_state, logits)

  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (_, (new_model_state, logits)), grad = compute_gradient_fn(train_state.params)

  if use_pmap:
    grad = jax.lax.pmean(grad, axis_name='batch')

  if config.get('max_grad_norm') is not None:
    grad = clip_grads(grad, config.max_grad_norm)

  tx = train_state.tx
  if tx is None:
    raise ValueError('train_state.tx, the Gradient Transformation, is None')

  updates, new_opt_state = tx.update(grad, train_state.opt_state, train_state.params)
  new_params = optax.apply_updates(train_state.params, updates)

  training_logs['l2_grads'] = jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)]))
  ps = jax.tree_util.tree_leaves(new_params)
  training_logs['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
  us = jax.tree_util.tree_leaves(updates)
  training_logs['l2_updates'] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
  training_logs['learning_rate'] = lr_fn(train_state.global_step)

  metrics = metrics_fn(logits, batch)

  new_train_state = train_state.replace(
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng)

  return new_train_state, metrics, training_logs


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    debug: Optional[bool] = False,
) -> Tuple[Dict[str, Tuple[float, int]], jnp.ndarray]:
  """Runs a single eval step."""
  variables = {'params': train_state.params, **train_state.model_state}
  logits = flax_model.apply(
      variables, batch['inputs'], train=False, mutable=False, debug=debug)
  metrics = metrics_fn(logits, batch)
  return metrics, logits


def _get_and_reduce_metrics(metrics, use_pmap: bool):
  if use_pmap:
    return jax.tree_util.tree_map(train_utils.unreplicate_and_get, metrics)
  return jax.device_get(metrics)


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Main training loop."""
  lead_host = jax.process_index() == 0
  model = model_cls(config, dataset.meta_data)

  rng, init_rng = jax.random.split(rng)
  params, model_state, num_trainable_params, gflops = train_utils.initialize_model(
      model_def=model.flax_model,
      input_spec=[(
          dataset.meta_data['input_shape'],
          dataset.meta_data.get('input_dtype', jnp.float32),
      )],
      config=config,
      rngs=init_rng)

  lr_fn = lr_schedules.get_learning_rate_fn(config)
  optimizer_config = optimizers.get_optax_optimizer_config(config)
  tx = optimizers.get_optimizer(optimizer_config, lr_fn, params=params)
  accumulation_steps = int(config.get('accumulation_steps', 1))
  if accumulation_steps > 1:
    tx = optax.MultiSteps(tx, every_k_schedule=accumulation_steps)
    logging.info('Using gradient accumulation: every_k=%d', accumulation_steps)

  opt_state = jax.jit(tx.init, backend='cpu')(params)
  rng, train_rng = jax.random.split(rng)
  chrono = train_utils.Chrono()

  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()})

  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(workdir, train_state)
  chrono.load(train_state.metadata['chrono'])

  if start_step == 0 and config.get('pretrain_checkpoint') is not None:
    checkpoint_dir = config.pretrain_checkpoint.get('checkpoint_path')
    restored_model_cfg = config.get('model')
    restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
        checkpoint_dir, train_state, assert_exist=True)
    train_state = model.init_from_train_state(
        train_state, restored_train_state, restored_model_cfg)
    del restored_train_state
    logging.info('Restored model from pretrain checkpoint at %s.', checkpoint_dir)
  elif start_step == 0 and config.get('init_from') is not None:
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
    if checkpoint_format == 'scenic':
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
    elif checkpoint_format == 'big_vision':
      restored_train_state = pretrain_utils.convert_big_vision_to_scenic_checkpoint(
          init_checkpoint_path, train_state)
      restored_model_cfg = copy.deepcopy(config)
      restored_model_cfg.model.classifier = config.init_from.get('classifier_type', 'token')
    else:
      raise ValueError(f'Unsupported checkpoint_format: {checkpoint_format}')

    train_state = model.init_from_train_state(
        train_state, restored_train_state, restored_model_cfg)
    del restored_train_state
    logging.info('Restored model from %s checkpoint at %s.', checkpoint_format,
                 init_checkpoint_path)

  train_state = train_state.replace(metadata={})

  use_pmap = jax.local_device_count() > 1
  if use_pmap:
    logging.info('Using pmap over %d local devices.', jax.local_device_count())
    train_state = jax_utils.replicate(train_state)
  else:
    logging.info('Using single-device jit path (optimized for one GPU).')

  total_steps, steps_per_epoch = train_utils.get_num_training_steps(config, dataset.meta_data)

  train_step_fn = functools.partial(
      train_step,
      flax_model=model.flax_model,
      loss_fn=model.loss_function,
      lr_fn=lr_fn,
      metrics_fn=model.get_metrics_fn('train'),
      config=config,
      debug=config.debug_train,
      use_pmap=use_pmap)
  eval_step_fn = functools.partial(
      eval_step,
      flax_model=model.flax_model,
      metrics_fn=model.get_metrics_fn('validation'),
      debug=config.debug_eval)

  if use_pmap:
    train_step_pmapped = jax.pmap(
        train_step_fn, axis_name='batch', donate_argnums=(0, 1))
    eval_step_pmapped = jax.pmap(
        eval_step_fn, axis_name='batch', donate_argnums=(1,))
  else:
    train_step_pmapped = jax.jit(train_step_fn, donate_argnums=(0, 1))
    eval_step_pmapped = jax.jit(eval_step_fn, donate_argnums=(1,))

  log_eval_steps = config.get('log_eval_steps') or steps_per_epoch
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  max_checkpoint_keep = config.get('max_checkpoint_keep', 3)
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps

  eval_batch_size = config.get('eval_batch_size', config.batch_size)
  total_eval_steps = int(np.ceil(dataset.meta_data['num_eval_examples'] / eval_batch_size))
  steps_per_eval = config.get('steps_per_eval') or total_eval_steps

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps,
      writer=writer,
      every_secs=None,
      every_steps=config.get('report_progress_step', log_summary_steps),
  )

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
      train_state, t_metrics, t_logs = train_step_pmapped(train_state, train_batch)
      train_metrics.append(t_metrics)
      extra_training_logs.append(_get_and_reduce_metrics(t_logs, use_pmap))

    for hook in hooks:
      hook(step)

    if ((step % log_summary_steps == 0) or (step == total_steps) or
        (lead_host and chrono.warmup)):
      chrono.pause(wait_for=(train_metrics,))
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              lambda x: _get_and_reduce_metrics(x, use_pmap), train_metrics),
          extra_training_logs=jax.tree_util.tree_map(jax.device_get, extra_training_logs),
          writer=writer)
      train_metrics, extra_training_logs = [], []
      chrono.resume()

    if (step % log_eval_steps == 0) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params,))
      with report_progress.timed('eval'):
        eval_metrics = []
        if use_pmap:
          train_state = train_utils.sync_model_state_across_replicas(train_state)
        for _ in range(steps_per_eval):
          eval_batch = next(dataset.valid_iter)
          e_metrics, _ = eval_step_pmapped(train_state, eval_batch)
          eval_metrics.append(_get_and_reduce_metrics(e_metrics, use_pmap))
        eval_summary = train_utils.log_eval_summary(
            step=step, eval_metrics=eval_metrics, writer=writer)
      writer.flush()
      del eval_metrics
      chrono.resume()

    if (((step % checkpoint_steps == 0) and (step > 0)) or (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        train_utils.handle_checkpointing(
            train_state, chrono, workdir, max_checkpoint_keep)
      chrono.resume()

  train_utils.barrier_across_hosts()
  assert train_summary is not None
  assert eval_summary is not None
  return train_state, train_summary, eval_summary
