import tensorflow as tf
import data_utils 
import numpy as np
import os
from pathlib import Path
from functools import partial
from functools import lru_cache
import audio_transform
import inspect

from tqdm.auto import tqdm

print(f"tf.__version__:{tf.__version__}")

@lru_cache(1)
def load_shuffled_ds_and_meta(seed=228922):
  ds,meta = data_utils.load_TFRecord()
  
  shuffled_ds = ds.shuffle(
    buffer_size = meta["total_files"],
    seed=228922,
    reshuffle_each_iteration=False
  )
  
  return shuffled_ds, meta

def train_valid_test_split(ds, valid_fold, total_folds, LEN):
  assert valid_fold >= 0 and valid_fold < total_folds and total_folds >= 3
  data = {
    "train":[],
    "valid":[],
    "test" :[], 
  }
  linspace = np.linspace(0, LEN,num=total_folds+1,dtype=np.int64)
  for start in range(total_folds):
    which_part = ["train", "valid", "test"][
      (start == valid_fold) + 2*( (start-1)%total_folds == valid_fold)
    ]
    data[which_part].append(
      ds.skip(linspace[start])
        .take(linspace[start+1]-linspace[start])
    )
    
  for k, v in data.items():
    base = v[0]
    for sub in v[1:]:
      base = base.concatenate(sub)
    data[k] = base
  return data


def use_items(*args, use):
  ith_table = {
    "filepath":0,
    "x":1,
    "x_pitch_shift":2,
    "mel":3,
    "pitch_shifted_mel":4,
    "label":5,
    "remark":6,
  }
  ret = []
  for name in use:
    ret.append(args[ith_table[name]])
    if "mel" in name:
      perm = [*range(len(ret[-1].shape))]
      perm[-1], perm[-2] = perm[-2], perm[-1]
      ret[-1] = tf.transpose(ret[-1],perm=perm)
  return ret

def random_pick_one(*args, shift_cap=None, mid_prob=0.9):
  assert args[0].shape[0]%2==1
  mid = args[0].shape[0]//2
  if shift_cap == None:
    shift_cap = mid
  else:
    assert shift_cap <= mid
  
  if tf.random.uniform([1]) > mid_prob:
    idx = tf.argmax(tf.random.uniform([2*shift_cap+1]))+(mid-shift_cap)
  else:
    idx = tf.constant(mid,dtype=tf.int64)
  return [args[0][idx], *args[1:]]

def process_train_valid_test_data_dict(
  data,
  use_shifted_mel=True,
  use_remark=True,
  **shift_kwargs):
  
  if use_remark:
    target_name = "remark"
  else:
    target_name = "label"
    
  use_Mel_Label = partial(use_items, use=["mel", target_name])

  train_ds = data["train"]
  valid_ds = data["valid"]
  
  train_ds = (
    train_ds
    .shuffle(99999, reshuffle_each_iteration=True)
  )
  
  if use_shifted_mel:
    use_PitchShifted_Mel_Label = partial(use_items, use=["pitch_shifted_mel", target_name])
    random_pick_shift = partial(random_pick_one, shift_cap=shift_kwargs.get("n", 5))
    train_ds = (
      train_ds
      .map(use_PitchShifted_Mel_Label)
      .map(random_pick_shift)
    )
  else:
    train_ds = train_ds.map(use_Mel_Label)
  
  def expand(*sample):
    return [tf.expand_dims(x,0) for x in sample]
  valid_ds = (
    valid_ds
    .map(use_Mel_Label)
  )
  
  if use_remark:
    mapper = lambda remark:tf.py_function(
      func=remark_mapper,
      inp=[remark], Tout=tf.int64
    )
    train_ds = train_ds.map(lambda x,remark:(x, mapper(remark)))
    valid_ds = valid_ds.map(lambda x,remark:(x, mapper(remark)))
  
  valid_ds = valid_ds.map(expand)
  return train_ds, valid_ds


def dynamic_batch_size(ds, batch_size_top=32, batch_size_low=1):
  out_raw = []
  batch_size = 0
  def process_raw():
    return tuple(map(lambda x:tf.stack(x,0),zip(*out_raw)))
  
  for sample in ds:
    batch_size = batch_size or np.random.randint(batch_size_low,batch_size_top+1)
    out_raw.append(sample)
    if len(out_raw) == batch_size:
      batch_size = 0
      yield process_raw()
      out_raw = []

  if out_raw:
    yield process_raw()
    

def create_dynamic_batchsize_ds(ds, batch_size_top, batch_size_low):
  assert batch_size_top >= batch_size_low
  if batch_size_top > batch_size_low:
    generator = lambda ds:partial(dynamic_batch_size,ds,batch_size_top=batch_size_top,batch_size_low=batch_size_low)
    out_ds = tf.data.Dataset.from_generator(
      generator(ds),
      output_types=(tf.float32, tf.float32),
    )
    return out_ds
  else:
    return ds.batch(batch_size_top)

def load_default_config():
  config = {
    "seed":228922,
    "total_folds":12,
    "cache":True,
    "use_shifted_mel":True,
    "use_remark":False,
    "shift_mel_kwargs":{"n":5},
    "batch_size_top":32,
    "batch_size_low":1,
  }
  return config

def config_helper():
  print(inspect.getsource(config_helper))
  print("="*30)
  print(inspect.getsource(load_default_config))

def lazy_ds_loader(**config):
  base_config = load_default_config()
  if config:
    base_config.update(config)
  config = base_config
    
  shuffled_ds, meta = load_shuffled_ds_and_meta(config.get("seed"))
  if config.get("cache"):
    shuffled_ds = shuffled_ds.cache()
  ds_multiple_folds = [
    train_valid_test_split(
      shuffled_ds,
      valid_fold,
      config.get("total_folds"),
      LEN=meta["total_files"]
    )
    for valid_fold in range(config.get("total_folds"))
  ]
  
  ds_multiple_folds = [
    process_train_valid_test_data_dict(
      data = data,
      use_remark=config.get("use_remark", False),
      **config.get("shift_mel_kwargs")
    )
    for data in ds_multiple_folds
  ]
  
  ds_multiple_folds = [
    [
      create_dynamic_batchsize_ds(
        train_ds,
        batch_size_top=config.get("batch_size_top"),
        batch_size_low=config.get("batch_size_low"),
      ),
      valid_ds]
    for train_ds, valid_ds in ds_multiple_folds
  ]
  
  return ds_multiple_folds, config, shuffled_ds

def remark_mapper(remark):
  return (
    {
      b'Barking': 0,
      b'Howling': 1,
      b'Crying': 2,
      b'COSmoke': 3,
      b'GlassBreaking': 4,
      b'Other': 6,
      b'Vacuum': 5,
      b'Blender': 6,
      b'Electrics': 6,
      b'Cat': 7,
      b'Dishes': 8,
    }[remark.numpy()]
  )