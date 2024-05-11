from __future__ import absolute_import, print_function


from collections import namedtuple
import argparse, json, os, requests, sys, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

import numpy as np
# from matplotlib import pyplot as plt

from collections import namedtuple
import tvm
from tvm import relay
from tvm.relay import quantize as qtz
from tvm.contrib import download
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir

import mxnet as mx
from mxnet import gluon
import logging
import os
import time

import logging
logging.basicConfig(level=logging.INFO)


Config = namedtuple('Config', ['model', 'nbit_input',  'dtype_input', 'nbit_output', 'dtype_output', 'global_scale', 'batch_size'])

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 2
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def get_model(model_name, batch_size, qconfig, target=None, original=False, simulated=False):
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    img_size = 299 if model_name == 'inceptionv3' else 224
    input_shape = (batch_size, 3, img_size, img_size)
    
    
    import torch
    import torchvision

    import time
    model_name = "resnet50"
    model = getattr(torchvision.models, model_name)(pretrained=True)


    model = model.eval()

    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    
    input_name = "input0"
    shape_list = [(input_name, (1, 3, 224, 224))]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


    # mod, params = relay.frontend.from_mxnet(gluon_model, {"data": input_shape})
    # net = mod['main']

    start_time = time.time()
    with relay.build_config(opt_level=3):
        qfunc = relay.quantize.prerequisite_optimize(mod, params=params)
    logging.debug('original')
    logging.debug(qfunc.astext(show_meta_data=False))
    if original:
        return qfunc

    with qconfig:
        logging.debug('current quantize config')
        logging.debug(qtz.current_qconfig())
        qfunc = qtz.quantize(qfunc)
        logging.debug('after quantize')
        logging.debug(qfunc.astext(show_meta_data=False))

    build_time = time.time() - start_time
    logging.info(model_name + " inference graph build in {0:.2f}s".format(build_time))


    return qfunc, params, input_shape


###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
               
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    else:
        open(tmp_log_file, 'a').close()

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.
def tune_and_evaluate(tuning_opt, cfg, target, ctx, log_file):
    qconfig = qtz.qconfig(skip_conv_layers=[0],
                        nbit_input=cfg.nbit_input,
                        nbit_weight=cfg.nbit_input,
                        global_scale=cfg.global_scale,
                        dtype_input=cfg.dtype_input,
                        dtype_weight=cfg.dtype_input,
                        dtype_activation=cfg.dtype_output,
                        debug_enabled_ops=None)

    # extract workloads from relay program
    logging.info("Extract tasks...")
    mod, params, input_shape = get_model(cfg.model, cfg.batch_size, qconfig, target)

    tasks = autotvm.task.extract_from_program(mod, target=target,
                                            params=params, ops=(relay.op.nn.conv2d,))
    for i in range(len(tasks)):
        op_name = tasks[i].workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            print ("Tuning {} is not supported on x86")
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        print ( "[Create Task %2d/%2d (%s, %s) ] " % (i+1, len(tasks), tasks[i].name, tasks[i].workload[0]))

        tsk = autotvm.task.create(func_create, args=tasks[i].args,
                                    target=tasks[i].target, template_key='direct')
        tsk.workload = tasks[i].workload
        tasks[i] = tsk

    # run tuning tasks
    logging.info("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        logging.info("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype('float32'))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        logging.info("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=60)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        logging.info("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))


if __name__ == "__main__":

    target = 'llvm -mcpu=skylake-avx512'
    ctx = tvm.cpu()

    configs = [
        Config('resnet18_v1', nbit_input=8, dtype_input='int8', nbit_output=8, dtype_output='int8', global_scale=8.0, batch_size=1),
        Config('resnet18_v1', nbit_input=16, dtype_input='int16', nbit_output=16, dtype_output='int16', global_scale=8.0, batch_size=1),
        # Config('mobilenetv2_1.0', nbit_input=8, dtype_input='int8', nbit_output=8, dtype_output='int8', global_scale=4.0, batch_size=1),
        # Config('mobilenetv2_1.0', nbit_input=16, dtype_input='int16', nbit_output=16, dtype_output='int16', global_scale=4.0, batch_size=1),
    ]

    for config in configs:
        logging.info('Start testing for %s', config.model)

        log_file = "%s_%s.log" % (config.model, config.dtype_input)
        if os.path.exists(log_file):
            os.remove(log_file)

        #### TUNING OPTION ####
        tuning_option = {
            'log_filename': log_file,

            'tuner': 'random',
            'n_trial': 10,
            'early_stopping': None,

            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
                # runner=autotvm.RPCRunner(
                #     '1080ti',  # change the device key to your key
                #     '0.0.0.0', 9190,
                #     number=20, repeat=3, timeout=4, min_repeat_ms=150)
            ),
        }

        tune_and_evaluate(tuning_option, config, target, ctx, log_file)