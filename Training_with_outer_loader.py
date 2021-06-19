import argparse
import os
import dataset_loader as loader
from pprint import PrettyPrinter
from tqdm.auto import tqdm
pprint = PrettyPrinter(indent=2).pprint
from datetime import datetime
import matplotlib.pyplot as plt

from dotmap import DotMap

import tensorflow as tf
# import data_utils 
import numpy as np
import os
from pathlib import Path
from functools import partial
from collections import defaultdict
import json


def my_dropout(x, drop_rate, gaussian_drop_rate):
  x = tf.keras.layers.GaussianDropout(gaussian_drop_rate)(x)
  LEN = len(x.shape)
  power = 1/LEN
  each_drop_rate = 1-(1-drop_rate)**power
  for channel in range(1,LEN):
    noise_shape = [None]*LEN
    noise_shape[channel] = 1
    x = tf.keras.layers.Dropout(each_drop_rate,noise_shape=noise_shape)(x)
  x = tf.keras.layers.Dropout(each_drop_rate,noise_shape=[None]*LEN)(x)
  return x

def conv_block_1d(x,filters,kernel_size,drop_rate, gaussian_drop_rate,activation,**conv_kwargs):
  assert len(x.shape) == 3, f"only support dims==3 for conv1d, got {len(x.shape)}"
  x = my_dropout(x, drop_rate,gaussian_drop_rate)
  conv = tf.keras.layers.Conv1D(
    filters, kernel_size,
    padding="same", **conv_kwargs)
  x = conv(x)
  x = activation(x)
  return x




def build_model(
  depth, filters,
  use_dilate,
  activation,
  global_kernel_size,
  drop_rate,
  gaussian_drop_rate,
  concat_mode,
  final_group_mode,
  input_shape=(216, 128),
):
  global categories
  
  layer_params = [
    [filters, [1, [2**power,]][use_dilate]] for power in range(depth)
  ]
  
  batch_shape = [None, *input_shape]
  conv_block = conv_block_1d
  
  Input = tf.keras.layers.Input(batch_shape=batch_shape)
  nodes = [Input]
  
  for filters, dilation_rate in layer_params:
    if concat_mode in ["None", None]:
      inputs = nodes[-1]
    elif concat_mode == "last":
      inputs = tf.concat(nodes[-2:], axis=-1)
    elif concat_mode == "all":
      inputs = tf.concat(nodes, axis=-1)
    elif concat_mode == "residual":
      if len(nodes) >= 2 and nodes[-1].shape[-1] == nodes[-2].shape[-1]:
        inputs = nodes[-1]+nodes[-2]
      else:
        inputs = nodes[-1]
    else:
      raise
    
    nodes.append(
      conv_block(
        inputs,
        filters,
        global_kernel_size,
        drop_rate=drop_rate,
        gaussian_drop_rate=gaussian_drop_rate,
        activation=activation,
        dilation_rate=dilation_rate,
      )
    )
  
  for hidden in [categories]:
    nodes.append(
      tf.keras.layers.Dense(
        hidden,
      )(nodes[-1])
    )
  
  final_ = []
  if "max" in final_group_mode:
    final_.append(tf.keras.layers.GlobalMaxPool1D()(nodes[-1]))
  if "mean" in final_group_mode:
    final_.append(tf.keras.layers.GlobalAveragePooling1D()(
      tf.keras.layers.Dropout(drop_rate)(nodes[-1])
    ))
  if "attention" in final_group_mode:
    if "last" in final_group_mode:
      inputs = nodes[-2] # before to categories
    elif "all" in final_group_mode:
      inputs = tf.concat(nodes, axis=-1)
    elif "first_and_last" in final_group_mode:
      inputs = tf.concat([nodes[0], nodes[-2]], axis=-1)
    
    hidden_neurons = filters
    before_porb_hidden = tf.keras.layers.Dense(hidden_neurons, activation="relu")(inputs)
    before_prob = tf.keras.layers.Dense(1, activation="relu")(before_porb_hidden/(inputs.shape[-1]**.5))
    prob = tf.keras.layers.Softmax(axis=-2)(before_prob)
    final_.append(
      tf.reduce_sum(
        prob*nodes[-1], axis=-2
      )
    )
    
  nodes.append(tf.reduce_mean(final_, axis=0))
  
  model = tf.keras.Model(
    inputs=nodes[0],
    outputs=nodes[-1]
  )
  
  return model


def get_initial_model(**model_config):
  model = build_model(
    **model_config
  )
  return model



def mix_up_on(ds):
  for _1, _2 in zip(ds, ds):
    r = np.random.beta(0.1, 0.1,tf.shape(_1[0])[0]).astype(np.float32)
    r0 = tf.reshape(r, [-1,1,1])
    r1 = tf.reshape(r, [-1,1])
    
    yield _1[0]*r0+_2[0]*(1-r0), _1[1]*r1+_2[1]*(1-r1)

    
def make_input_shape_right(*args):
  global categories
  assert len(args)==2
  a0,a1 = args
  a1 = tf.one_hot(a1.numpy().astype(np.int64), categories)
  return a0, a1


def start_training_same_batch_size(ith_fold, ds_folds_list, model_folder, big_config):
  (train_ds, valid_ds)  = ds_folds_list[ith_fold]
  
  model_config = big_config.model_config
  other_config = big_config.other_config
  
  train_ds = train_ds.map(
    lambda x,y: tf.py_function(
      func=make_input_shape_right,
      inp=[x,y],
      Tout=(tf.float32,tf.float32)
    )
  )
  
  valid_ds = valid_ds.map(
    lambda x,y : tf.py_function(func=make_input_shape_right,
      inp=[x,y],
      Tout=(tf.float32,tf.float32)
    )
  )
  
  opt = tf.keras.optimizers.Adam(other_config.lr)
  if other_config.multilabel_problem:
    loss = tf.keras.losses.BinaryCrossentropy(
      from_logits=True,
      name="loss",
    )
    metrics = [tf.keras.metrics.BinaryAccuracy(name="bin_acc")]
  else:
    loss = tf.keras.losses.CategoricalCrossentropy(
      from_logits=True,
      name="loss",
    )
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
  
  model = get_initial_model(**big_config.model_config)
  model.compile(
    optimizer=opt,
    loss=loss,
    metrics=metrics,
  )
  
  patient = 100
  monitor="loss" # early_stop will use val_XXX version
  
  base_model_path = model_folder+f"/model.valid_{ith_fold:02d}"
  last_model_path = base_model_path+".last.h5"
  best_val_model_path = base_model_path+".best_val.h5"
  best_train_model_path = base_model_path+".best_train.h5"
  history_path = model_folder+f"/history.valid_{ith_fold:02d}.json"
  
  cbs = [
    # best_val_model
    tf.keras.callbacks.ModelCheckpoint(
      filepath=best_val_model_path, monitor="val_"+monitor, verbose=1, save_best_only=True,
    ),
    # best_val_model
    tf.keras.callbacks.ModelCheckpoint(
      filepath=best_train_model_path, monitor=monitor, verbose=1, save_best_only=True,
    ),
    # last model
    tf.keras.callbacks.ModelCheckpoint(
      filepath=best_train_model_path, verbose=0,
    ),
    # early_stop with val 
    tf.keras.callbacks.EarlyStopping(
      monitor="val_"+monitor, patience=patient, verbose=1,
    )
  ]
  if other_config.mixup:
    train_ds = tf.data.Dataset.from_generator(
      partial(mix_up_on,train_ds),
      (tf.float32, tf.float32)
    )
  all_history = defaultdict(list)
  print(datetime.time(datetime.now()))
  history = model.fit(
    train_ds.prefetch(5),
    epochs=other_config.epoches,
    validation_data=valid_ds.prefetch(5),
    verbose=2,
    callbacks=cbs
  )
  for k, v in history.history.items():
    all_history[k]+=v
    
  keys, lists = zip(*all_history.items())
  best_ith_epoch = np.argmin(all_history["val_"+monitor])
  print(f"best_epoch : {best_ith_epoch}")
  print(keys, ", (", *map(lambda l:f"{l[best_ith_epoch]:.3f}", lists), ")")
    
  with open(history_path,"w") as f:
    json.dump(all_history,f)
    
def lazy_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('--force', type=int, help="If set it to True, it will overwrite", required=False, default=0)
  
  parser.add_argument('--config_helper', type=bool, help="If set it to True, it will print default model's config", required=False, default=False)

  parser.add_argument('--model_folder', type=str, required=False, help='Directory of save Model', default="TEST_MODEL")
  # model_config
  parser.add_argument('--depth', type=int, required=False, default=5)
  parser.add_argument('--filters', type=int, required=False, default=256)
  parser.add_argument('--global_kernel_size', type=int, required=False, default=4)
  parser.add_argument('--use_dilate', type=bool, required=False, default=True)
  parser.add_argument('--drop_rate', type=float, required=False, default=.3)
  parser.add_argument('--concat_mode', type=str, required=False, default="None")
  parser.add_argument('--gaussian_drop_rate', type=float, required=False, default=0.05)
  parser.add_argument('--activation', type=str, required=False, default="relu")
  parser.add_argument('--final_group_mode', type=str, required=False, default="mean,max")
  
  # other_config
  parser.add_argument('--lr', type=float, required=False, default=0.0001)
  parser.add_argument('--use_remark', type=bool, required=False, default=False)
  parser.add_argument('--multilabel_problem', type=int, required=False, default=1)
  parser.add_argument('--total_folds', type=int, required=False, default=12)
  parser.add_argument('--epoches', type=int, required=False, default=1000)
  parser.add_argument('--mixup', type=int, required=False, default=1)
  parser.add_argument('--batch_size_low', type=int, required=False, default=32)
  parser.add_argument('--batch_size_top', type=int, required=False, default=32)
  
  
  args = parser.parse_args()
  
  if args.config_helper:
    pprint(loader.config_helper())
    exit(0)
  
  model_folder = args.model_folder

  if not args.force:
    assert not os.path.exists(model_folder), (
      "Folder is exist "
      "if want to overwrite, please set "
      "--force=True\n"
      f"Folder Path : {model_folder}"
    )
  
  model_config = DotMap({
    "depth":args.depth,
    "filters":args.filters,
    "global_kernel_size":args.global_kernel_size,
    "use_dilate":args.use_dilate,
    "drop_rate":args.drop_rate,
    "concat_mode":args.concat_mode,
    "gaussian_drop_rate":args.gaussian_drop_rate,
    "activation":vars(tf.keras.activations)[args.activation],
    "final_group_mode":args.final_group_mode.split(","),
  })

  other_config = DotMap({
    "lr":args.lr,
    "use_remark":args.use_remark,
    "multilabel_problem":args.multilabel_problem,
    "total_folds":args.total_folds,
    "epoches":args.epoches,
    "mixup":args.mixup,
    "batch_size_low":args.batch_size_low,
    "batch_size_top":args.batch_size_top,
  })

  big_config = DotMap({
    "model_config":model_config,
    "other_config":other_config,
  })
  
  return model_folder, big_config
  
if __name__ == "__main__":
  model_folder, big_config = lazy_parser()
  Path(model_folder).mkdir(exist_ok=True,parents=True)
  
  model_config = big_config.model_config
  other_config = big_config.other_config
  json.dump({k:str(v) for k,v in dict(model_config).items()}, open(model_folder+"/model_config.json","w"))
  json.dump({k:str(v) for k,v in dict(other_config).items()}, open(model_folder+"/other_config.json","w"))
  
  pprint(big_config)
  pprint(big_config.model_config.activation.__name__)
  
  global categories
  if big_config.other_config.use_remark:
    categories=9
  else:
    categories=6
  
  example_model = get_initial_model(**big_config.model_config)

  example_model.summary()
  del example_model

  ds_folds_list, ds_config, base_ds = loader.lazy_ds_loader(
    total_folds=other_config.total_folds,
    use_remark=other_config.use_remark,
    batch_size_low=other_config.batch_size_low,
    batch_size_top=other_config.batch_size_top,
  )
  
  pprint(ds_config)
  
  D = {}
  if ds_config["cache"]:
    print("initital cached ds")
    for _ in tqdm(base_ds):
      rrr=_[-1].numpy()
      if rrr not in D:
        D[rrr]=0
      D[rrr]+=1
      continue
    pprint(D)
    
  for ith_fold in range(other_config.total_folds):
    start_training_same_batch_size(ith_fold, ds_folds_list, model_folder, big_config)
  