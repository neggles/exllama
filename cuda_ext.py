# from abc import ABC
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import (
    Tensor,
    version as torch_version,
)
from torch.utils.cpp_extension import load

library_dir = Path(__file__).parent.absolute()
extension_name = "exllama_ext"
verbose = False

ext_root = library_dir.joinpath(extension_name)

# another kludge to get things compiling in Windows
windows = sys.platform == "win32"
if windows:
    vs_versions = ["2015", "2017", "2019", "2022"]
    vs_editions = ["Community", "Professional", "Enterprise", "BuildTools", "Preview"]
    programfiles_dirs = [
        os.environ.get("ProgramFiles", None),
        os.environ.get("ProgramFiles(x86)", None),
        os.environ.get("ProgramW6432", None),
    ]
    msvc_search_paths = [
        Path(f"{pf_dir}\\Microsoft Visual Studio\\{version}\\{edition}\\VC\\Tools\\MSVC\\")
        for version in vs_versions
        for pf_dir in programfiles_dirs
        for edition in vs_editions
    ]
    msvc_search_paths = [x for x in msvc_search_paths if x.exists() and x.is_dir()]

    def find_msvc() -> Path | None:
        for msvc_dir in msvc_search_paths:
            compiler_versions = sorted([x for x in msvc_dir.iterdir() if x.is_dir()], reverse=True)
            for cver in compiler_versions:
                compiler_path: Path = cver.joinpath("bin", "Hostx64", "x64", "cl.exe")
                if compiler_path.exists() and compiler_path.is_file():
                    return compiler_path.parent
        return None  # didn't find it :(

    import subprocess

    res = subprocess.run(["where.exe", "/Q", "cl"], shell=True)
    if res.returncode != 0:
        # not in path, try to find and inject it
        cl_path = find_msvc()
        if cl_path is not None:
            print(f"Found MSVC compiler at {cl_path}")
            os.environ["path"] += f";{cl_path}"
        else:
            raise RuntimeError("Unable to find MSVC compiler; giving up...")

# extra linker flags
extra_ldflags = list()
if sys.platform == "win32":
    extra_ldflags.append("cublas.lib")
    if sys.base_prefix != sys.prefix:
        extra_ldflags.append(f"/LIBPATH:{Path(sys.base_prefix).joinpath('libs')}")

# extra compiler flags
extra_cuda_cflags = ["-lineinfo"]
if torch_version.hip is not None:
    extra_cuda_cflags.extend(["-U__HIP_NO_HALF_CONVERSIONS__", "-O3"])


exllama_ext: ModuleType = load(
    name=extension_name,
    sources=[
        f"{ext_root}/exllama_ext.cpp",
        f"{ext_root}/cuda_buffers.cu",
        f"{ext_root}/cuda_func/q4_matrix.cu",
        f"{ext_root}/cuda_func/q4_matmul.cu",
        f"{ext_root}/cuda_func/column_remap.cu",
        f"{ext_root}/cuda_func/rms_norm.cu",
        f"{ext_root}/cuda_func/rope.cu",
        f"{ext_root}/cuda_func/half_matmul.cu",
        f"{ext_root}/cuda_func/q4_attn.cu",
        f"{ext_root}/cuda_func/q4_mlp.cu",
        f"{ext_root}/cpu_func/rep_penalty.cpp",
    ],
    extra_include_paths=[f"{ext_root}"],
    verbose=verbose,
    extra_ldflags=extra_ldflags,
    extra_cuda_cflags=extra_cuda_cflags,
    extra_cflags=["-O3"]
    # extra_cflags = ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
)

# from exllama_ext import set_tuning_params
# from exllama_ext import prepare_buffers
# from exllama_ext import q4_mlp
from exllama_ext import (  # noqa: E402
    apply_rep_penalty,
    half_matmul,
    half_matmul_cublas,
    make_q4,
    q4_matmul,
    q4_matmul_lora,
    rep_penalty,
    rms_norm,
    rope_,
)

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension
none_tensor = torch.empty((1, 1), device="meta")


def ext_make_q4(qweight, qzeros, scales, g_idx, device):
    """Construct Q4Matrix, return handle"""
    return make_q4(qweight, qzeros, scales, g_idx if g_idx is not None else none_tensor, device)


def ext_q4_matmul(x, q4, q4_width, lora_A=None, lora_B=None):
    """Matrix multiplication, returns x @ q4"""
    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype=torch.float16, device=x.device)

    if lora_A is None:
        q4_matmul(x, q4, output)
    else:
        lora_temp = torch.empty((x.shape[0], lora_A.shape[1]), dtype=torch.float16, device=x.device)
        q4_matmul_lora(x, q4, output, lora_A, lora_B, lora_temp)

    return output.view(outshape)


def ext_half_matmul(x, w, cublas=False) -> Tensor:
    """Matrix multiplication, returns x @ w, both half-precision tensors"""
    outshape = x.shape[:-1] + (w.shape[1],)
    x = x.view(-1, x.shape[-1])

    if cublas:
        output = torch.empty((x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul_cublas(x, w, output)
    else:
        output = torch.zeros((x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul(x, w, output)

    return output.view(outshape)


def ext_rope_(x, sin, cos, past_len, num_heads, head_dim):
    """RoPE embeddings, in_place"""
    rope_(x, sin, cos, past_len, num_heads, head_dim)


def ext_rms_norm(x, w, epsilon):
    """RMS norm: x = x * w / sqrt(row_mean(x * x) + epsilon)"""
    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    output = torch.empty_like(x)
    rms_norm(x, w, output, epsilon)

    return output.view(outshape)


def ext_rms_norm_(x, w, epsilon):
    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    rms_norm(x, w, x, epsilon)


def ext_rep_penalty_mask_cpu(vocab_size, sequence, penalty_max, sustain, decay):
    """Repetition penalty"""
    rep_mask = torch.empty(vocab_size, dtype=torch.float32)
    rep_penalty(sequence, rep_mask, penalty_max, sustain, decay)
    return rep_mask


def ext_apply_rep_penalty_mask_cpu(sequence, penalty_max, sustain, decay, logits):
    apply_rep_penalty(sequence, penalty_max, sustain, decay, logits)
