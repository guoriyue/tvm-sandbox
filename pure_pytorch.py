import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
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


# from PIL import Image

# img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
# img_path = download_testdata(img_url, "cat.png", module="data")
# img = Image.open(img_path).resize((224, 224))

# # Preprocess the image and convert to tensor
# from torchvision import transforms

# my_preprocess = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# img = my_preprocess(img)
# img = np.expand_dims(img, 0)


input_name = "input0"
shape_list = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


# _node_defaults = {
#         "nbit_input": 8,
#         "nbit_weight": 8,
#         "nbit_activation": 32,
#         "dtype_input": "int8",
#         "dtype_weight": "int8",
#         "dtype_activation": "int32",
#         "calibrate_mode": "global_scale",
#         "global_scale": 8.0,
#         "weight_scale": "power2",
#         "skip_dense_layer": True,
#         "skip_conv_layers": [0],
#         "do_simulation": False,
#         "round_for_shift": True,
#         "debug_enabled_ops": None,
#         "rounding": "UPWARD",
#         "calibrate_chunk_by": -1,
#         "partition_conversions": "disabled",
#     }

# # quantization
# with relay.quantize.qconfig(calibrate_mode="global_scale",
#                             global_scale=8.0,
#                             nbit_activation=16,
#                             dtype_activation="int16",
#                             skip_conv_layers=[],
#                             skip_dense_layer=False,
#                             partition_conversions="enabled",
#                             do_simulation=False):
#     mod = relay.quantize.quantize(mod, params)



target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)


# from tvm.contrib import graph_executor

# dtype = "float32"
# m = graph_executor.GraphModule(lib["default"](dev))
# before_tvm_inference = time.time()
# # Set inputs
# m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
# # Execute
# m.run()
# # Get outputs
# tvm_output = m.get_output(0)
# after_tvm_inference = time.time()
# print("tvm_inference", after_tvm_inference - before_tvm_inference)


import mxnet as mx


calibration_rec = download_testdata(
    "http://data.mxnet.io.s3-website-us-west-1.amazonaws.com/data/val_256_q90.rec",
    "val_256_q90.rec",
)

batch_size = 64

def get_val_data(num_workers=4):
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    def batch_fn(batch):
        return batch.data[0].asnumpy(), batch.label[0].asnumpy()
    img_size = 299 if model_name == "inceptionv3" else 224
    val_data = mx.io.ImageRecordIter(
        path_imgrec=calibration_rec,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,
        resize=256,
        data_shape=(3, img_size, img_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return val_data, batch_fn

validation_samples = 64 * 50
val_data, batch_fn = get_val_data()
val_data.reset()
# from tvm.contrib import graph_executor

# # for quantization
# dtype = "float32"
# # tvm_model = relay.create_executor("vm", mod, dev, target).evaluate()
# tvm_model = graph_executor.GraphModule(lib["default"](dev))
# sum_tvm_inference = 0
# top1_acc = 0
# top5_acc = 0
# batch_size = 64
# for i, batch in enumerate(val_data):
#     data, target = batch_fn(batch)
#     for j in range(batch_size):
#         before_tvm_inference = time.time()
#         tvm_model.set_input(input_name, tvm.nd.array(data[j].reshape((1, 3, 224, 224)).astype(dtype)))
#         # Execute
#         tvm_model.run()
#         # Get outputs
#         tvm_output = tvm_model.get_output(0)
#         # tvm_output = tvm_model(data[j].reshape((1, 3, 224, 224)).astype(dtype))
#         after_tvm_inference = time.time()
#         # print(after_tvm_inference - before_tvm_inference)
#         sum_tvm_inference += after_tvm_inference - before_tvm_inference

#         tvm_output = tvm_output.asnumpy()
#         top1 = np.argsort(tvm_output, axis = 1)[:,-1:]
#         top5 = np.argsort(tvm_output, axis = 1)[:,-5:]
#         if target[j] in top1:
#             top1_acc += 1
#         if target[j] in top5:
#             top5_acc += 1
#     print("batch tvm_inference", sum_tvm_inference/(i+1))
#     print("batch tvm_accuracy top1_acc", top1_acc/(i+1))
#     print("batch tvm_accuracy top5_acc", top5_acc/(i+1))
#     # top1_acc += np.sum(np.array([1 if target[k] in top1[k] else 0 for k in range(len(top1))]))
#     # top5_acc += np.sum(np.array([1 if target[k] in top5[k] else 0 for k in range(len(top5))]))

# print("tvm_inference", sum_tvm_inference/validation_samples)
# print("tvm_accuracy top1_acc", top1_acc/validation_samples)
# print("tvm_accuracy top5_acc", top5_acc/validation_samples)

# synset_url = "".join(
#     [
#         "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_synsets.txt",
#     ]
# )
# synset_name = "imagenet_synsets.txt"
# synset_path = download_testdata(synset_url, synset_name, module="data")
# with open(synset_path) as f:
#     synsets = f.readlines()

# synsets = [x.strip() for x in synsets]
# splits = [line.split(" ") for line in synsets]
# key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

# class_url = "".join(
#     [
#         "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_classes.txt",
#     ]
# )
# class_name = "imagenet_classes.txt"
# class_path = download_testdata(class_url, class_name, module="data")
# with open(class_path) as f:
#     class_id_to_key = f.readlines()

# class_id_to_key = [x.strip() for x in class_id_to_key]



# # Get top-1 result for TVM
# top1_tvm = np.argmax(tvm_output.numpy()[0])
# tvm_class_key = class_id_to_key[top1_tvm]
validation_samples_cnt = 0
# # Get top-1 result for quantized PyTorch
# tvm_top1_torch = np.argmax(tvm_output.numpy())
# tvm_torch_class_key = class_id_to_key[tvm_top1_torch]
sum_torch_inference = 0
top1_acc = 0
top5_acc = 0
for i, batch in enumerate(val_data):
    # Convert input to PyTorch variable and get PyTorch result for comparison
    with torch.no_grad():
        data, target = batch_fn(batch)
        for j in range(batch_size):
            validation_samples_cnt += 1
            torch_data = torch.from_numpy(data[j])
            before_torch_inference = time.time()
            output = model(torch_data.reshape((1, 3, 224, 224)))
            after_torch_inference = time.time()
            # print(after_torch_inference - before_torch_inference)
            sum_torch_inference += after_torch_inference - before_torch_inference
            
            output = output.numpy()
            top1 = np.argsort(output, axis = 1)[:,-1:]
            top5 = np.argsort(output, axis = 1)[:,-5:]
            if target[j] in top1:
                top1_acc += 1
            if target[j] in top5:
                top5_acc += 1
            # top1_acc += np.sum(np.array([1 if target[k] in top1[k] else 0 for k in range(len(top1))]))
            # top5_acc += np.sum(np.array([1 if target[k] in top5[k] else 0 for k in range(len(top5))]))
        print("batch torch_inference", sum_torch_inference/validation_samples_cnt)
        print("batch torch_accuracy top1_acc", top1_acc/validation_samples_cnt)
        print("batch torch_accuracy top5_acc", top5_acc/validation_samples_cnt)
        # # Get top-1 result for PyTorch
        # top1_torch = np.argmax(output.numpy())
        # torch_class_key = class_id_to_key[top1_torch]
print("torch_inference", sum_torch_inference/validation_samples_cnt)
print("torch_accuracy top1_acc", top1_acc/validation_samples_cnt)
print("torch_accuracy top5_acc", top5_acc/validation_samples_cnt)







# # print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
# print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
# print("Quantized Torch top-1 id: {}, class name: {}".format(tvm_top1_torch, key_to_classname[tvm_torch_class_key]))

# import tvm
# from tvm import relay

# import numpy as np

# from tvm.contrib.download import download_testdata

# # PyTorch imports
# import torch
# import torchvision

# import time
# model_name = "resnet50"
# model = getattr(torchvision.models, model_name)(pretrained=True)


# model = model.eval()

# # We grab the TorchScripted model via tracing
# input_shape = [1, 3, 224, 224]
# input_data = torch.randn(input_shape)
# scripted_model = torch.jit.trace(model, input_data).eval()


# from PIL import Image

# img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
# img_path = download_testdata(img_url, "cat.png", module="data")
# img = Image.open(img_path).resize((224, 224))

# # Preprocess the image and convert to tensor
# from torchvision import transforms

# my_preprocess = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )
# img = my_preprocess(img)
# img = np.expand_dims(img, 0)


# input_name = "input0"
# shape_list = [(input_name, img.shape)]
# mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)




# # # quantization
# # with relay.quantize.qconfig(calibrate_mode="global_scale",
# #                             global_scale=8.0,
# #                             nbit_activation=16,
# #                             dtype_activation="int16",
# #                             skip_conv_layers=[],
# #                             skip_dense_layer=False,
# #                             partition_conversions="enabled"):
# #     mod = relay.quantize.quantize(mod, params)



# target = tvm.target.Target("llvm", host="llvm")
# dev = tvm.cpu(0)
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, params=params)


# from tvm.contrib import graph_executor

# dtype = "float32"
# m = graph_executor.GraphModule(lib["default"](dev))
# sum_tvm_inference = 0
# for i in range(10):
#     before_tvm_inference = time.time()
#     # Set inputs
#     m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
#     # Execute
#     m.run()
#     # Get outputs
#     tvm_output = m.get_output(0)
#     after_tvm_inference = time.time()
#     print(after_tvm_inference - before_tvm_inference)
#     sum_tvm_inference += after_tvm_inference - before_tvm_inference
# print("tvm_inference", sum_tvm_inference/10)


# # # for quantization
# # dtype = "float32"
# # quant_model = relay.create_executor("vm", mod, dev, target).evaluate()

# # before_quant_inference = time.time()
# # quant_output = quant_model(img)
# # # quant_model.set_input(input_name, tvm.nd.array(img.astype(dtype)))
# # # quant_model.run()
# # # quant_model.get_output(0)
# # after_quant_inference = time.time()
# # print("quant_inference", after_quant_inference - before_quant_inference)


# synset_url = "".join(
#     [
#         "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_synsets.txt",
#     ]
# )
# synset_name = "imagenet_synsets.txt"
# synset_path = download_testdata(synset_url, synset_name, module="data")
# with open(synset_path) as f:
#     synsets = f.readlines()

# synsets = [x.strip() for x in synsets]
# splits = [line.split(" ") for line in synsets]
# key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

# class_url = "".join(
#     [
#         "https://raw.githubusercontent.com/Cadene/",
#         "pretrained-models.pytorch/master/data/",
#         "imagenet_classes.txt",
#     ]
# )
# class_name = "imagenet_classes.txt"
# class_path = download_testdata(class_url, class_name, module="data")
# with open(class_path) as f:
#     class_id_to_key = f.readlines()

# class_id_to_key = [x.strip() for x in class_id_to_key]



# # Get top-1 result for TVM
# top1_tvm = np.argmax(tvm_output.numpy()[0])
# tvm_class_key = class_id_to_key[top1_tvm]

# # # Get top-1 result for quantized PyTorch
# # quant_top1_torch = np.argmax(quant_output.numpy())
# # quant_torch_class_key = class_id_to_key[quant_top1_torch]

# sum_torch_inference = 0
# # Convert input to PyTorch variable and get PyTorch result for comparison
# for i in range(10):
#     # Convert input to PyTorch variable and get PyTorch result for comparison
#     with torch.no_grad():
#         torch_img = torch.from_numpy(img)
#         before_torch_inference = time.time()
#         output = model(torch_img)
#         after_torch_inference = time.time()
#         print(after_torch_inference - before_torch_inference)
#         sum_torch_inference += after_torch_inference - before_torch_inference
        


#         # Get top-1 result for PyTorch
#         top1_torch = np.argmax(output.numpy())
#         torch_class_key = class_id_to_key[top1_torch]
# print("torch_inference", sum_torch_inference/10)







# print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
# print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
# # print("Quantized Torch top-1 id: {}, class name: {}".format(quant_top1_torch, key_to_classname[quant_torch_class_key]))




