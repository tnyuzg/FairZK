# FAIRZK: A Scalable System to Prove Machine Learning Fairness in Zero-Knowledge

## Overview

This repository contains the code for FAIRZK, accepted at IEEE S&P 2025.

FairZK designs a specialized SNARK protocol to prove the fairness bound of a DNN model from its parameters and some aggregated information of the input without doing real inference.

## Setup

1. **Install Rust**: Follow the instructions on [Rust Installation](https://www.rust-lang.org/tools/install).
2. **Verify Installation**: Post-installation, ensure everything is set up correctly with:

   ```bash
   cargo --version
   rustup --version
   ```
3. **Use the Nightly Toolchain**:

   ```bash
   rustup default nightly
   ```

## Run the Code

To run the example `/fairzk/examples/total.rs`, which is a 3-layer DNN on Adult dataset:

```bash
cargo run --example total --release
```

expected output example:

```
row_num: 128, col_num: 37
row_num: 128, col_num: 128
row_num: 1, col_num: 128
row_num: 45222, col_num: 37
```

which presents shapes of weight matrices

```
construct lookup instance: 13 ms
commit oracles: 81 ms
commit lookup first oracles: 81 ms
commit lookup second oracles: 79 ms
prove infairence sumcheck: 0 ms
prove spectral norm sumcheck: 2 ms
prove l2 norm sumcheck: 0 ms
prove bound sumcheck: 0 ms
prove total snark lookup sumcheck: 29 ms
prove oracles: 253 ms

== Prove Timing (in ms) ==
Construct:          13 ms
Commit Oracles:     241 ms
Prove Oracles:      253 ms
Prove Sumcheck:     31 ms
```

which presents the components of the prover time

```

verify infairence sumcheck: 0 ms
verify spectral norm sumcheck: 0 ms
verify l2 norm sumcheck: 0 ms
verify bound sumcheck: 0 ms
verify lookup sumcheck: 0 ms
verify oracle: 184 ms

== Verify Timing (in ms) ==
Verify Oracles:     184 ms
Verify Sumcheck:    0 ms
```

which presents the components of the verifier time

```
proof size total 349 MB
```

which presents the proof size

```
total success
```

which presents whether the verifier accepts the proof

``fairzk/sumcheck`` contains sumcheck protocol taken from [arkwork](https://github.com/arkworks-rs/sumcheck).


``fairzk/piop`` contains PIOPs for different components, such as spectral norm. ``fairzk/piop/total`` combines all the components.


``fairzk/snark`` contains usable SNARKs by compiling PIOPs with PCS.


``models`` contains quantized parameters of multiple models. You can modify the file paths in `/fairzk/examples/total.rs` to evaluate the peformance on different models.
