import torch
try:
    import torch_npu
except ImportError:
    print('Please install torch_npu!')

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Conv1d(80, 1024, kernel_size=3, padding=1)

    def load_ckpt(self, weight_path, bias_path):
        self.model.weight = torch.nn.Parameter(torch.load(weight_path, map_location="cpu")).bfloat16()
        self.model.bias = torch.nn.Parameter(torch.load(bias_path, map_location="cpu")).bfloat16()

    def forward(self, x):
        return self.model(x)

class Model2(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Linear(1024, 1024)

    def load_ckpt(self, weight_path, bias_path):
        self.model.weight = torch.nn.Parameter(torch.load(weight_path, map_location="cpu")).bfloat16()
        self.model.bias = torch.nn.Parameter(torch.load(bias_path, map_location="cpu")).bfloat16()

    def forward(self, x):
        return self.model(x)


def compute_on_device(device="cpu"):
    model = Model()
    model.load_ckpt(
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/gpu/step0/rank0/dump_tensor_data/Functional.conv1d.0.forward.input.1.pt",
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/gpu/step0/rank0/dump_tensor_data/Functional.conv1d.0.forward.input.2.pt"
    )
    model.to(device)
    input = torch.load(
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/gpu/step0/rank0/dump_tensor_data/Functional.conv1d.0.forward.input.0.pt", map_location="cpu"
    ).to(device).bfloat16()
    saved_output = torch.load(  # cpu forward 和报存的npu
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/gpu/step0/rank0/dump_tensor_data/Functional.conv1d.0.forward.output.0.pt"
    ).to(device)
    output = model(input).bfloat16()
    print(output.dtype)
    print(saved_output.dtype)
    print(torch.allclose(output, saved_output))
    return output, saved_output
def compute_on_device2(device="cpu"):
    model = Model2()
    model.load_ckpt(
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/npu/step0/rank0/dump_tensor_data/Functional.linear.3.forward.input.1.pt",
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/npu/step0/rank0/dump_tensor_data/Functional.linear.3.forward.input.2.pt"
    )
    model.to(device)
    input = torch.load(
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/npu/step0/rank0/dump_tensor_data/Functional.linear.3.forward.input.0.pt", map_location="cpu"
    ).to(device).bfloat16()
    saved_output = torch.load(  # cpu forward 和报存的npu
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/npu/step0/rank0/dump_tensor_data/Functional.linear.3.forward.output.0.pt"
    ).to(device)
    output = model(input).bfloat16()
    print(output.dtype)
    print(saved_output.dtype)
    print(torch.allclose(output, saved_output))
    return output, saved_output

def compute_err(act, golden):
    act, golden = act.flatten(), golden.flatten()
    err = act - golden
    abs_error = err.abs()
    rel_err = abs_error / golden.abs()
    print("千分之一", torch.sum(rel_err < 1e-3) / rel_err.numel())

if __name__ == "__main__":
    # torch_npu.npu.conv.allow_hf32 = False
    # cpu_out, cpu_saved_out = compute_on_device(device="cpu")
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    # )
    # with torch_npu.profiler.profile(
    #     activities=[
    #         torch_npu.profiler.ProfilerActivity.CPU,
    #         torch_npu.profiler.ProfilerActivity.NPU
    #         ],
    #     schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./check_conv"),
    #     record_shapes=True,
    #     with_stack=True,
    #     with_flops=False,
    #     with_modules=False,
    #     experimental_config=experimental_config) as prof:
    #         for step in range(1):
    #             npu_out, npu_saved_out = compute_on_device(device="npu")
    #             prof.step()
    npu_out, npu_saved_out = compute_on_device2(device="npu")
    # npu_out, npu_saved_out = compute_on_device(device="cpu")
    npu_out = npu_out.cpu()
    npu_saved_out = npu_saved_out.cpu()

    saved_output = torch.load(
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/npu/step0/rank0/dump_tensor_data/Functional.linear.3.forward.output.0.pt", map_location="cpu"
    ).cpu()
    saved_output_gpu = torch.load(
        f"/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/dump/gpu/step0/rank0/dump_tensor_data/Functional.linear.3.forward.output.0.pt",
        map_location="cpu"
    ).cpu()
    # print(torch.allclose(
    #     cpu_out,
    #     npu_out
    # ))
    # print(torch.allclose(
    #     cpu_saved_out,
    #     npu_saved_out
    # ))
    compute_err(npu_out, saved_output)
    compute_err(saved_output_gpu, saved_output)
    compute_err(npu_out, saved_output_gpu)