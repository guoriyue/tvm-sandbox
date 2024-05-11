"""
AutoTensorize: find mappings for a conv2d layer
===========================================
**Author**: `Size Zheng <https://github.com/KnowingNothing>`


In this tutorial, AMOS will find mappings for conv2d
by fully exploring the mapping space.
The target ISA is AVX-512.
AMOS will jointly explore both mappings and schedules
at the same time.
"""

# AMOS/tutorials/auto_tensorize/mapping_ops/explore_mapping_conv2d_u8s8s32_avx512.py
import tvm
import os
from tvm import auto_tensorize as at


def conv2d(N, C, H, W, K, R, S, stride, padding):
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="uint8", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="int8", name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: tvm.te.sum(
            (
                Pad[n, rc, p * stride + rr, q * stride + rs].astype("int32")
                * B[k, rc, rr, rs].astype("int32")
            ),
            axis=[rc, rr, rs],
        ),
        name="Conv",
    )
    return [A, B, Conv]


def tensorize_avx512_u8s8s32(N, C, H, W, K, R, S, stride, padding, layer):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding)
    target_dag = at.compute_dag_from_tensors([Conv])
    # target = "llvm -mcpu=skylake-avx512"
    target = "llvm -mcpu=cortex-a72"

    log_dir = "conv2d-u8s8s32-layer-%s" % (layer)
    log_file = "conv2d-u8s8s32-layer-%s.log" % (layer)

    trials = 1000
    measure_opt = at.MeasureOptions(
        # target_host="llvm -mcpu=skylake-avx512",
        target_host= "llvm -mcpu=cortex-a72",
        target=target,
        timeout=100,
        number=200,
        min_repeat_ms=500,
    )

    result = at.auto_tensorize_v4(
        target_dag,
        target,
        log_file,
        measure_opt,
        schedule_log_dir=log_dir,
        trials=trials,
        search_group_size=10,
        transform_dump=False,
        build_parallel=1,
        run_parallel=1,
    )
    # result = at.auto_tensorize(
    #     target_dag, target, log_file, measure_opt,
    #     trials=trials, search_group_size=16,
    #     verbose=False, transform_dump=False)
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

    # load from file
    # schedule_gen.load_from_file(log_file, clear=True)
    # entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = result.params, result.perf
    print(value)
    print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)
    return cost


def run(N, C, H, W, K, R, S, stride, padding, layer):
    return tensorize_avx512_u8s8s32(N, C, H, W, K, R, S, stride, padding, layer)


yolo_shapes_b1 = [
    # yolo
    (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
    (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    # # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
    # # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]


res18_shapes_b1 = [
    # resnet-18
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]


if __name__ == "__main__":
    batches = [2 ** i for i in range(1)]
    beg = 0
    num = 15
    for batch in batches:
        costs = []
        for i, shape in enumerate(res18_shapes_b1[beg : beg + num]):
            (_, C, H, W, K, _, R, S, _, stride, padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            cost = run(
                N,
                C,
                H,
                W,
                K,
                R,
                S,
                stride,
                padding,
                f"({N},{C},{H},{W},{K},{R},{S},{stride},{padding})",
            )
            costs.append(cost)
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)