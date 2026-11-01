# Initial FP32 combined optimization check: failed

2026-10-23, `openmm_dev`, CPU, two threads, acetamide leg start frame, POLAR-1-M and operating head.
Command: `check_native.py --threads 2 --repeats 1 --dtype float32 --qs 1.5,2.5 --out native_cpu_fp32_01.json`
(the actual output argument included this experiment directory).

The first case, legacy path q=1.5, failed the predeclared force-equivalence threshold of
0.05 kJ/mol/A. Maximum force difference was 0.09390259 kJ/mol/A; 3 of 7,902 force components
exceeded the threshold. Reference and candidate used the same positions and weights; the
candidate combined scalar propagation and batched fragment energies.

The initial harness asserted before writing its first case, so no `native_cpu_fp32_01.json`
was produced. This record preserves the failure rather than treating it as missing or passed.
The harness was then changed to save failed cases before raising and to isolate scalar-only
and batch-only variants. The threshold was not relaxed.
