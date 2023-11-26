# MlirSynth: Automatic Program Raising in MLIR using Program Synthesis

## Overview
MlirSynth is a compiler approach for automatically raising lower-level MLIR dialects to higher-level ones without manually defined transformation rules. It leverages program synthesis, type constraints, and equivalences to explore and identify optimal transformations. MlirSynth enables retargetability to new MLIR dialects and domain-specific accelerators, unlocking significant performance improvements.

## Features
- **Automatic Raising**: Translates low-level MLIR dialects to high-level ones without predefined raising rules.
- **Multi-Dialect Support**: Capable of targeting multiple MLIR dialects, including Linalg and HLO.
- **Program Synthesis**: Employs bottom-up enumerative synthesis guided by type constraints and observational equivalence.
- **Performance Improvements**: Achieves significant speedup on CPUs, and TPUs over standard LLVM-O3 compilation flows.

## Build Instructions
```sh
# Build dependencies
./build_tools/build_dependencies.sh

# Build MlirSynth
./build_tools/build_mlirSynth.sh
```

## Usage
### Running MlirSynth
To raise an MLIR program to the high-level StableHLO dialect, use the following command. The raised program will be printed to stdout.
```sh
./build/synthesizer input.mlir --target-dialect=hlo --guide
```

## Benchmarking
MlirSynth has been evaluated on the **Polybench benchmark suite**, showing superior coverage and performance compared to state-of-the-art methods.
The following script will run all benchmarks:
```sh
python ./benchmark/run.py
```

## Citation
```bibtex
@inproceedings{brauckmann2023mlirsynth,
  title={mlirsynth: Automatic, retargetable program raising in multi-level ir using program synthesis},
  author={Brauckmann, Alexander and Polgreen, Elizabeth and Grosser, Tobias and O'Boyle, Michael FP},
  booktitle={2023 32nd International Conference on Parallel Architectures and Compilation Techniques (PACT)},
  pages={39--50},
  year={2023},
  organization={IEEE}
}
```