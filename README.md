# TODO Tasks

## Base
1. Инференс лламы | **DONE**
2. Подмена линейных слоев на кастомные | **DONE**
## Benchmark
3. Замер времени инференса и перплексии | **DONE**
4. Сделать обработку инференса батчами
5. Квантование линейных слоев через pytorch (?)
6. Замер скорости операции MM W16 vs W4 (SeqLen: 128, 512, 2048, LinLayer: 2048x2048, 2048x512, 2048x8192, 2048x128256)
## Triton kernels
7. Global Quantization (per-tensor) + *(7.1)* упаковка
8. Rowwise Quantization + *(8.1)* упаковка
9. Matmul bf16 x int4
