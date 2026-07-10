from setuptools import setup, Extension

sources = [
    "bindings/module.c",
    "pointretta_src/src/tensor/tensor_alloc.c",
    "pointretta_src/src/tensor/tensor_ops.c",
    "pointretta_src/src/model/retention_multihead.c",
    "pointretta_src/src/model/retention_parallel.c",
    "pointretta_src/src/model/retention_recurrent.c",
    "pointretta_src/src/model/norm.c",
    "pointretta_src/src/model/ffn.c",
    "pointretta_src/src/model/model.c",
    "pointretta_src/src/autograd/backward_ops.c",
    "pointretta_src/src/train/model_train.c",
    "pointretta_src/src/train/loss.c",
    "pointretta_src/src/optimizer/adafactor.c",
    "pointretta_src/src/tokenizer/bpe_train.c",
    "pointretta_src/src/tokenizer/bpe_encode.c",
    "pointretta_src/src/format/pr_file.c",
    "pointretta_src/src/format/json_parser.c",
    "pointretta_src/src/kernels/dispatch.c",
]

ext = Extension(
    "pointretta._core",
    sources=sources,
    include_dirs=["pointretta_src/include"],
    extra_compile_args=["-O2"],
    libraries=["m"],
)

setup(ext_modules=[ext])
