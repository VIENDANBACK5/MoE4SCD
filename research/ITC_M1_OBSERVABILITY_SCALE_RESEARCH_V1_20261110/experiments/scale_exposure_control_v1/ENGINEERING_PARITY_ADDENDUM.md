# One numerical-engineering correction, before scientific training

The first disposable probe failed its strict pointwise bf16gradient tolerance at backbone.stem.conv1.weight, after primal losses passed. PROBE_FAILURE.json and probe.log are preserved. Neither scientific arm was started and no quality metrics were consulted.

Before another probe, lock this diagnostic: compare all available parameter gradients for plain/checkpointed forward in native1024first under fp32and then bf16, with identical weights, targets and RNG. Use full fp32to test recomputation semantics (loss relative1e-5, gradients atol1e-6/rtol1e-3). Measure bf16global relativeL2/cosine and per-tensor absolute differences rather than equating elementwise near-zero rounding with changed supervision.

Only permit the original2048engineering optimizer step if fp32parity passes and bf16global relativeL2<=.02and every non-negligible gradient tensor has cosine>=.99. Report the numerical differences explicitly. This replaces only the unsuitable bf16pointwise engineering assertion, not architecture, matching, trainable scope, data, loss, precision of scientific training, schedule or inference. The identical checkpointing strategy remains common to both arms. If this second bounded diagnostic fails, stop this execution path; no resolution reduction, module-freezing or further numerical-tolerance sweep.

Probe2weights/optimizer remain disposable and cannot initialize a scientific arm. Initial failure is not erased or relabeled as success. No model efficacy or original method follows from numerical parity.
