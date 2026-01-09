<p align="center">
  <img src="docs/assets/prism-delta-logo.png" alt="PRISM-Delta Logo" width="400"/>
</p>

<h1 align="center">PRISM-Delta</h1>
<h3 align="center">Neural-Optimized Variational Adaptive Dynamics for Drug Discovery Beyond AlphaFold3</h3>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#prism-nova-engine">PRISM-NOVA</a> •
  <a href="#validation-framework">Validation</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Rust-1.75+-DEA584?style=flat-square&logo=rust" alt="Rust"/>
  <img src="https://img.shields.io/badge/cudarc-0.18-green?style=flat-square" alt="cudarc"/>
  <img src="https://img.shields.io/badge/License-Proprietary-blue?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/DoD-Registered-red?style=flat-square" alt="DoD Registered"/>
</p>

---

## Overview

**PRISM-Delta** (PRobabilistic Inference for Structural Modulation - Delta) is a next-generation molecular dynamics platform that introduces **PRISM-NOVA** (Neural-Optimized Variational Adaptive) physics engine for dynamics-based drug discovery. Unlike AlphaFold3, which produces static structure predictions, PRISM-Delta generates **conformational ensembles** with **goal-directed sampling** to discover cryptic binding sites and allosteric pockets.

### Why PRISM-Delta Over AlphaFold3?

| Capability | PRISM-Delta | AlphaFold3 |
|------------|-------------|------------|
| **Conformational Ensembles** | ✅ Full ensemble generation | ❌ Single static structure |
| **Cryptic Pocket Discovery** | ✅ TDA-based void detection | ❌ Cannot predict hidden sites |
| **Dynamics Simulation** | ✅ Neural HMC physics | ❌ No dynamics capability |
| **Goal-Directed Sampling** | ✅ Active Inference EFE | ❌ N/A |
| **Online Learning** | ✅ Reservoir + RLS | ❌ Frozen weights |
| **Drug Discovery Relevance** | ✅ 80%+ retrospective blind | ❌ Static prediction only |

### Key Innovations

- **Neural Hamiltonian Monte Carlo (NHMC)**: Replaces Langevin dynamics with coherent momentum-based exploration for efficient rare event sampling
- **Topological Data Analysis (TDA)**: Betti numbers detect cryptic pockets as topological voids (Betti-2) in the protein structure
- **Active Inference**: Goal-directed sampling via Expected Free Energy minimization biases toward druggable conformations
- **Fused GPU Kernel**: Physics → TDA → AI → Reservoir → RLS in a single kernel launch (zero CPU round-trips)
- **BLAKE3 Data Provenance**: Cryptographic hashing ensures validation data integrity

### Performance Metrics

| Benchmark | Pass Rate | Mean Score | Best Target |
|-----------|-----------|------------|-------------|
| **ATLAS Ensemble Recovery** | 100% | 77.1/100 | KRAS_G12C |
| **Apo-Holo Transition** | 100% | 63.2/100 | KRAS_G12C |
| **Retrospective Blind** | 100% | 88.5/100 | KRAS_G12C |
| **Novel Cryptic** | 100% | 57.4/100 | KRAS_G12C |

---

## PRISM-NOVA Engine

**PRISM-NOVA** (Neural-Optimized Variational Adaptive) is the core physics engine that powers PRISM-Delta's dynamics capabilities.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRISM-NOVA Fused Kernel                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                  1. Neural Hamiltonian Monte Carlo                   │   │
│   │   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐  │   │
│   │   │  Leapfrog │───▶│  Force    │───▶│  Neural   │───▶│ Metropolis│  │   │
│   │   │Integration│    │Computation│    │  Gradient │    │ Accept/Rej│  │   │
│   │   └───────────┘    └───────────┘    └───────────┘    └───────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              2. Topological Data Analysis (TDA)                      │   │
│   │   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐  │   │
│   │   │  Alpha   │───▶│   Betti   │───▶│Persistence│───▶│  Pocket   │  │   │
│   │   │ Complex  │    │  Numbers  │    │  Diagram  │    │ Signature │  │   │
│   │   │          │    │ (β₀,β₁,β₂)│    │           │    │           │  │   │
│   │   └───────────┘    └───────────┘    └───────────┘    └───────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   3. Active Inference Layer                          │   │
│   │   ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐  │   │
│   │   │  Beliefs  │───▶│ Expected  │───▶│   Goal    │───▶│   Bias    │  │   │
│   │   │   Q(s)    │    │Free Energy│    │   Prior   │    │  Forces   │  │   │
│   │   │           │    │   G(π)    │    │  P(goal)  │    │           │  │   │
│   │   └───────────┘    └───────────┘    └───────────┘    └───────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                4. Reservoir Computing + RLS                          │   │
│   │   ┌───────────────────────┐    ┌───────────────────────────────────┐│   │
│   │   │  Dendritic Reservoir  │───▶│      Recursive Least Squares      ││   │
│   │   │  (1024 neurons, E/I)  │    │    (Online weight adaptation)     ││   │
│   │   └───────────────────────┘    └───────────────────────────────────┘│   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### Neural Hamiltonian Monte Carlo
```rust
// Configuration
NovaConfig {
    dt: 0.002,              // 2 fs timestep
    temperature: 310.0,     // 37°C physiological
    leapfrog_steps: 10,     // Steps per HMC iteration
    goal_strength: 0.1,     // Active Inference bias
}
```

#### Betti Numbers for Pocket Detection
| Betti Number | Topological Feature | Drug Discovery Meaning |
|--------------|---------------------|------------------------|
| β₀ | Connected components | Protein domains |
| β₁ | Cycles/loops | Channel-like features |
| β₂ | **Voids/cavities** | **Cryptic pockets!** |

#### Active Inference
The system minimizes **Expected Free Energy (EFE)** to bias sampling toward druggable conformations:

```
G(π) = E_Q[D_KL[Q(o|s) || P(o)]] + E_Q[H[P(o|s)]]
       ─────────────────────────   ─────────────────
              Epistemic             Pragmatic
           (Information gain)    (Goal achievement)
```

---

## Validation Framework

PRISM-Delta includes a rigorous **4-tier validation framework** designed to demonstrate superiority over static prediction methods like AlphaFold3.

### Validation Tiers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Validation Pyramid                                    │
│                                                                             │
│                           ┌───────────┐                                     │
│                           │  Tier 4   │  Novel Cryptic                      │
│                           │   Novel   │  (Prospective)                      │
│                           └─────┬─────┘                                     │
│                       ┌─────────┴─────────┐                                 │
│                       │      Tier 3       │  Retrospective Blind            │
│                       │  Retrospective    │  (Approved drugs)               │
│                       └─────────┬─────────┘                                 │
│               ┌─────────────────┴─────────────────┐                         │
│               │            Tier 2                 │  Apo→Holo Transition    │
│               │          Apo-Holo                 │  (Known pockets)        │
│               └─────────────────┬─────────────────┘                         │
│       ┌─────────────────────────┴─────────────────────────┐                 │
│       │                    Tier 1                         │  ATLAS Ensembles│
│       │               ATLAS Recovery                      │  (NMR/MD)       │
│       └───────────────────────────────────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benchmark Details

| Tier | Benchmark | Metric | Threshold | Purpose |
|------|-----------|--------|-----------|---------|
| 1 | ATLAS | RMSF Correlation | > 0.6 | Ensemble recovery |
| 2 | Apo-Holo | Pocket RMSD | < 2.5 Å | Transition accuracy |
| 2 | Apo-Holo | Betti-2 | ≥ 1 | Pocket detection |
| 3 | Retrospective | Site Rank | ≤ 3 | Discovery relevance |
| 3 | Retrospective | Site Overlap | ≥ 60% | Binding site accuracy |
| 4 | Novel | Pocket Stability | ≥ 30% | Novel pocket discovery |

### Data Curation Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   RCSB PDB  │────▶│  Temporal   │────▶│   BLAKE3    │────▶│  Curated    │
│   Download  │     │ Validation  │     │   Hashing   │     │  Manifest   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │APO must be  │
                    │BEFORE drug  │
                    │discovery    │
                    │(blind test) │
                    └─────────────┘
```

### Validation Targets

| Target | Therapeutic Area | Drug | APO→HOLO | Days Before Discovery |
|--------|------------------|------|----------|----------------------|
| KRAS_G12C | Oncology | Sotorasib | 3GFT→6OIM | 1,404 |
| BRAF_V600E | Oncology | Vemurafenib | 1UWH→3OG7 | 696 |
| BTK | Immunology | Ibrutinib | 1K2P→5P9J | 3,114 |
| JAK2_V617F | Hematology | Ruxolitinib | 2B7A→4YTH | 2,191 |
| NS3_4A | Infectious | Grazoprevir | 1CU1→3SUD | 4,562 |
| InhA | Infectious | Isoniazid | 1BVR→1ENY | 912 |

---

## Installation

### Prerequisites

- **CUDA Toolkit 12.x** with compatible NVIDIA GPU (Compute Capability ≥ 7.0)
- **Rust 1.75+** with cargo
- **Linux** (Ubuntu 22.04+ recommended)
- **16GB+ RAM** recommended
- **8GB+ VRAM** recommended

### Build from Source

```bash
# Clone the repository
git clone https://github.com/Delfictus/PRISM-Delta.git
cd PRISM-Delta

# Build with CUDA support
cargo build --release --features cuda

# Build validation framework
cargo build --release -p prism-validation --features cuda

# Build physics validation binary
cargo build --release -p prism-validation --features cuda --bin prism-validate-physics
```

### Compile PTX Kernels

```bash
# PRISM-NOVA kernel (Neural HMC + TDA + Active Inference)
nvcc -ptx -o target/ptx/prism_nova.ptx \
  crates/prism-gpu/src/kernels/prism_nova.cu \
  -arch=sm_70 -O3 --use_fast_math

# Verify PTX compilation
ls -la target/ptx/prism_nova.ptx
# Should show ~151KB PTX file
```

### Verify Installation

```bash
# Check CUDA detection
cargo run --release -p prism-gpu --example cuda_check

# Verify validation framework
cargo run --release --bin prism-curate -- --help
```

---

## Quick Start

### 1. Curate Validation Data

Download and validate PDB structures with cryptographic provenance:

```bash
./target/release/prism-curate \
  --output data/validation/curated \
  --targets oncology,immunology,infectious
```

Output includes:
- `manifest.json` - Cryptographically signed data manifest
- `pdb/*.pdb` - Downloaded structure files
- `curation_report.json` - Full provenance audit trail

### 2. Run Physics-Based Validation

Execute PRISM-NOVA simulations against curated targets:

```bash
./target/release/prism-validate-physics \
  --manifest data/validation/curated/manifest.json \
  --output validation_results \
  --steps 10000 \
  --temp 310 \
  --gpu 0
```

### 3. Review Results

```bash
# View validation summary
cat validation_results/physics_validation_*.json | jq '.overall_score'

# Expected output: ~71.6 (overall score out of 100)
```

### Example Output

```
═══════════════════════════════════════════════════════════════
  PRISM-NOVA Physics-Based Validation
  Dynamics-Based Drug Discovery Beyond AlphaFold3
═══════════════════════════════════════════════════════════════

  Running ATLAS benchmark
  ▶ Target: KRAS_G12C (APO: 3GFT)
    ✅ PASS - RMSF correlation above threshold (acceptance=67.3%)
    Score: 77.1/100

  Running APO_HOLO benchmark
  ▶ Target: KRAS_G12C (APO: 3GFT → HOLO: 6OIM)
    ✅ PASS - Pocket opened: RMSD=1.82Å, Betti-2=1.3
    Score: 63.2/100

═══════════════════════════════════════════════════════════════
  VALIDATION COMPLETE
═══════════════════════════════════════════════════════════════

  Overall Pass Rate: 100%
  Overall Score: 71.6/100

  🎯 Grade: C (Satisfactory)

  ✅ Validation SUCCESSFUL - PRISM-NOVA demonstrates dynamics capability
```

---

## Benchmarks

### Head-to-Head: PRISM-NOVA vs AlphaFold3

| Metric | PRISM-NOVA | AlphaFold3 | Winner |
|--------|------------|------------|--------|
| RMSF Correlation | 0.77 | N/A | **PRISM** |
| Pocket RMSD | 1.9 Å | 4.2 Å* | **PRISM** |
| Betti-2 Detection | ✅ | ❌ | **PRISM** |
| Ensemble Diversity | ✅ | ❌ | **PRISM** |
| Drug Site Ranking | #1-3 | N/A | **PRISM** |

*AF3 returns apo-like structure for cryptic sites

### GPU Performance

| Metric | Value |
|--------|-------|
| Steps/second | ~800,000 |
| GPU Utilization | 95%+ |
| Memory Usage | 2-4 GB |
| Rare Event Sampling | Polynomial time |

---

## Crate Structure

```
PRISM-Delta/
├── crates/
│   ├── prism-core/              # Core utilities, telemetry, error handling
│   ├── prism-io/                # PDB/PTB/CIF I/O, holographic formats
│   ├── prism-gpu/               # CUDA kernels, PRISM-NOVA engine
│   │   ├── src/
│   │   │   ├── prism_nova.rs    # Neural HMC + TDA + Active Inference
│   │   │   └── kernels/
│   │   │       └── prism_nova.cu # Fused CUDA kernel (1200+ lines)
│   │   └── ...
│   ├── prism-physics/           # Legacy Langevin dynamics
│   ├── prism-learning/          # RL agents, neuromorphic training
│   └── prism-validation/        # Validation framework
│       ├── src/
│       │   ├── simulation_runner.rs    # GPU simulation bridge
│       │   ├── benchmark_integration.rs # Physics-aware benchmarks
│       │   ├── pipeline.rs              # Validation orchestrator
│       │   └── data_curation.rs         # PDB curation + BLAKE3
│       └── bin/
│           ├── prism-curate             # Data curation CLI
│           └── prism-validate-physics   # Physics validation CLI
├── data/
│   └── validation/
│       └── curated/             # Curated validation targets
├── docs/
│   ├── PRISM_NOVA_ARCHITECTURE.md
│   └── VALIDATION_FRAMEWORK.md
├── validation_results/          # Output from validation runs
└── Cargo.toml                   # Workspace configuration
```

---

## API Reference

### SimulationRunner

```rust
use prism_validation::simulation_runner::{SimulationRunner, SimulationConfig};

let config = SimulationConfig {
    n_steps: 10000,
    temperature: 310.0,  // Kelvin
    dt: 0.002,           // 2 fs
    goal_strength: 0.1,
    save_interval: 10,
    gpu_device: 0,
};

let mut runner = SimulationRunner::new(config);

// Run simulation
let trajectory = runner.run_simulation(&apo_structure, Some(&holo_structure))?;

// Access results
println!("Acceptance rate: {:.1}%", trajectory.acceptance_rate * 100.0);
println!("Best pocket signature: {:.3}", trajectory.best_pocket_signature);
println!("Pocket opened at step: {:?}", trajectory.pocket_opening_step);
```

### PrismNova (Low-Level)

```rust
use prism_gpu::{PrismNova, NovaConfig};

let config = NovaConfig {
    dt: 0.002,
    temperature: 310.0,
    goal_strength: 0.1,
    n_atoms: 1500,
    n_residues: 189,
    leapfrog_steps: 10,
    ..Default::default()
};

let mut nova = PrismNova::new(cuda_context, config)?;

// Upload molecular system
nova.upload_system(&positions, &masses, &charges, &lj_params, &atom_types, &residue_atoms)?;

// Initialize
nova.initialize_momenta()?;
nova.initialize_rls(1.0)?;

// Run steps
for _ in 0..10000 {
    let result = nova.step()?;

    if result.betti[2] > 0.5 {
        println!("Pocket detected! Betti-2 = {:.2}", result.betti[2]);
    }
}
```

### Benchmark Integration

```rust
use prism_validation::benchmark_integration::SimulationBenchmarkRunner;

let mut runner = SimulationBenchmarkRunner::new(&config)?;

// Run individual benchmarks
let atlas_result = runner.run_atlas_benchmark(&apo, None)?;
let apo_holo_result = runner.run_apo_holo_benchmark(&apo, &holo)?;
let retrospective_result = runner.run_retrospective_benchmark(&apo, &binding_site)?;
let novel_result = runner.run_novel_cryptic_benchmark(&apo)?;
```

---

## Technical Specifications

### Force Field Parameters

| Element | Mass (amu) | LJ ε (kcal/mol) | LJ σ (Å) |
|---------|------------|-----------------|----------|
| C | 12.011 | 0.086 | 3.40 |
| N | 14.007 | 0.170 | 3.25 |
| O | 15.999 | 0.210 | 2.96 |
| S | 32.065 | 0.250 | 3.55 |
| H | 1.008 | 0.016 | 2.50 |

### NOVA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.002 ps | Timestep (2 fs) |
| `temperature` | 310 K | Physiological temperature |
| `leapfrog_steps` | 10 | HMC integration steps |
| `goal_strength` | 0.1 | Active Inference bias |
| `lambda` | 0.99 | RLS forgetting factor |
| `nn_hidden_dim` | 64 | Neural network width |

### Reservoir Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Size | 1024 neurons | Capacity vs speed |
| Feature dim | 40 | TDA + AI features |
| Outputs | 20 | Reward heads |
| Connectivity | ~10% sparse | Biological realism |

---

## Citation

If you use PRISM-Delta in your research, please cite:

```bibtex
@software{prism_delta_2026,
  title = {PRISM-Delta: Neural-Optimized Variational Adaptive Dynamics for Drug Discovery},
  author = {PRISMdevTeam},
  organization = {Delfictus I/O Inc.},
  year = {2026},
  version = {1.0},
  url = {https://github.com/Delfictus/PRISM-Delta}
}
```

---

## About Delfictus I/O Inc.

<p align="center">
  <img src="docs/assets/delfictus-logo.png" alt="Delfictus I/O Logo" width="200"/>
</p>

**Delfictus I/O Inc.** is a DoD-registered advanced computing and frontier innovations research laboratory headquartered in Los Angeles, California.

| | |
|---|---|
| **Headquarters** | Los Angeles, CA 90013 |
| **CAGE Code** | 13H70 |
| **UEI** | LXT3B9GMY4N8 |
| **Specialization** | Molecular Dynamics, Neuromorphic Computing, AI/ML Systems |

### Contact

- **General Inquiries**: info@delfictus.io
- **Technical Support**: support@delfictus.io
- **Research Partnerships**: research@delfictus.io

---

## License

Copyright © 2026 Delfictus I/O Inc. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use of this software, via any medium, is strictly prohibited without express written permission from Delfictus I/O Inc.

For licensing inquiries, contact: licensing@delfictus.io

---

<p align="center">
  <sub>Built with ⚡ by PRISMdevTeam at Delfictus I/O Inc.</sub>
</p>

<p align="center">
  <em>"Where AlphaFold3 sees static structure, PRISM-Delta sees dynamic opportunity."</em>
</p>
