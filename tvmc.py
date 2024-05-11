from tvm.driver import tvmc

# model = tvmc.load('/home/ubuntu/scripts/resnet50-v2-7.onnx', shape_dict={'data': (1, 3, 224, 224)}) #Step 1: Load the model (and convert to Relay
# package = tvmc.compile(model, target="llvm", package_path="/home/ubuntu/scripts/resnet50-v2-7-tvm-package") #Step 2: Compile the model

new_package = tvmc.TVMCPackage(package_path="/home/ubuntu/scripts/resnet50-v2-7-tvm-package")
result = tvmc.run(new_package, device="cpu") #Step 3: Run
print(result)
# result = tvmc.run(package, device="cpu") #Step 3: Run


# tvmc.tune(model, target="llvm", enable_autoscheduler = True) #Step 1.5: Optional Tune

# package = tvmc.compile(model, target="llvm", package_path="/home/ubuntu/scripts/resnet50-v2-7-tvm-tuned-package") #Step 2: Compile the model

# result = tvmc.run(package, device="cpu") #Step 3: Run