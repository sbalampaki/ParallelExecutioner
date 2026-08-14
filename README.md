# ParallelExecutioner: Pre-Execution Parallelisation Advice for Heavy-Tailed Workloads

An execution advisor that predicts which parallelisation optimisations are worthwhile *before* running anything. Reads the CSR row-offset array—a single array of integers—and returns three things: which of three performance regimes a given partition count falls in, the ceiling on what any contiguous partitioner could recover there, and the worker count beyond which no assignment of whole rows can help.

**No kernel, no machine, no timing, no prior run.** This is the distinguishing property: existing advisors learn from scheduler accounting logs and therefore require the job to have run before, while this reads metadata about the data to be processed.

## The Workload: ICU Feature Extraction on MIMIC-IV

The workload studied here is feature extraction on clinical ICU data:

- **Dataset**: MIMIC-IV (Medical Information Mart for Intensive Care)
- **Scale**: 67,218 patient stays, 156.9 million timestamped events
- **Output**: 67,218 × 160 feature matrix  
- **Skew**: Records per stay are severely non-uniform (Gini 0.528; largest stay holds 103× the median)
- **Frequency of recomputation**: The extraction pass is rebuilt whenever the cohort, observation window, or variable set changes—so execution histories cannot accumulate

This realistic, heavy-tailed sparse workload is why existing predict-after-execution advisors break down.

## Key Findings

### Prediction Accuracy

**Imbalance computation from offsets alone is exact**: Predicted imbalance matches measurement to within 1.5×10⁻⁴ % at ten distinct (partitioner, partition-count) points.

**The model reproduces kernel behaviour**: Where a one-stage and a two-stage partitioner cut predict different values, measurement selects the two-stage value in all six test cases—so the model captures how work is *divided*, not just how much is assigned.

### Three Regimes

The prediction identifies three regimes, with both boundaries computed rather than measured:

| Regime | Range | Bottleneck | Recommendation |
|--------|-------|-----------|-----------------|
| **I: Placement-limited** | p < ~56 | Thread placement dominates (32.0% loss against at most 4.7% for any partitioner) | Fix thread placement before touching the partitioner |
| **II: Partition-limited** | ~56 ≤ p < p_atom | Partitioner is the lever; gain grows with p | Use a contiguous, work-balanced partitioner |
| **III: Atomicity-limited** | p ≥ p_atom | Mean partition < largest indivisible row | Only sub-unit decomposition escapes |

Where **p_atom = N / max(nᵢ)** is the worker count at which the mean partition equals the largest row.

### A Placement Law that Standard Metrics Miss

Throughput falls **8–10% per doubling** of contiguous same-socket thread groups (at fixed imbalance), cumulating to **24.3%**. This level shift—aggregate loss equals the median thread's loss to within 0.3 percentage points—is invisible to max-of-threads metrics.

### Shape, Not Size

Across twelve CSR structures from one cohort:
- **Vary record count 4.54×** → ceiling moves only 1.2×  
- **Hold record count fixed, vary distribution** → ceiling moves **19.5×**

Advice follows shape, not magnitude.

### Microarchitecture Transfer

Across four Intel server generations (12 years apart), six of eight effects hold their sign while platform ratios reach 4.4×. Sign transfers; magnitude doesn't.

## Repository Structure

```
ParallelExecutioner/
├── executionAdvisor/
│   ├── pstar_curve.py              # Core prediction model
│   ├── code/
│   │   ├── skewcast.py             # User-facing advisor CLI
│   │   ├── csr.py                  # CSR data structure and I/O
│   │   ├── cohort.py               # MIMIC-IV cohort definition
│   │   ├── convert.py              # Data format conversion
│   │   ├── kernel_*.c              # Parallel kernels (serial, OMP, Pthreads, MPI)
│   │   ├── validate.py             # Bit-identity verification
│   │   ├── ml_validate.py          # Machine-learning validation
│   │   ├── e0_distribution.py      # Distribution analysis
│   │   ├── e3_*.sbatch             # Experiment sweep jobs
│   │   ├── e4_*.sbatch             # Validation and scaling jobs
│   │   ├── variants.json           # Workload variant configurations
│   │   └── [build scripts, analysis tools, probes, common utilities]
│   └── results/
│       ├── manifest/               # Experiment metadata
│       └── variants/               # Results by variant
└── README.md
```

## Usage

### Quick Start: Analyse a Workload

Given a binary file of int64 CSR row offsets:

```bash
./executionAdvisor/code/skewcast.py --offsets <path> --cores <worker_count>
```

Example output:

```
============================================================================
ADVICE AT p = 32
============================================================================
  allocation              32 workers (~2.0 nodes at 16 cores/node)
  regime                  II -- partition-limited
  partitioner ceiling     24.53%   (2.92 placement rungs)
  placement span          32.0%   (per-node, does not grow with p)

  >> The partitioner is now the lever, and its value grows with p.
     1. Use a contiguous, work-balanced partitioner.
     2. Avoid non-contiguous assignments under a two-stage (rank-then-thread) decomposition.
     3. Choose the node count so the unit count divides evenly across first-stage cuts.
     4. Keep thread placement fixed -- its cost has not gone away, it has merely been overtaken.
```

### Workload Summary Statistics

Analyse a workload given only summary statistics (n_units, n_records, max_unit):

```bash
./executionAdvisor/code/skewcast.py \
  --n-records 156900000 \
  --max-stay 103000 \
  --n-stays 67218 \
  --sweep 16,32,64,96
```

Note: p* (crossover threshold) is *transferred* from the study cohort when full offsets unavailable; p_atom and distributional summaries are exact.

### Batch Comparison

Compare multiple offset arrays:

```bash
./executionAdvisor/code/skewcast.py \
  --batch csr/W1/offsets.i64 csr/W2/offsets.i64 \
  --json results.json
```

## The Model: Two-Stage Partitioning

The core insight: partition imbalance can be computed purely from the row-offset cumsum, before any execution.

### Block Partitioning

Index-balanced (block) cut: divide the index range [0, n) evenly.

```python
def block_imbalance(csum, n, R, T):
    """Two-stage floor-division cut."""
    total = csum[n]
    # First stage: R ranks
    rb = (np.arange(R + 1) * n) // R
    # Second stage: T threads per rank
    m = rb[1:] - rb[:-1]
    tt = np.arange(T + 1)
    b = rb[:-1][:, None] + (tt[None, :] * m[:, None]) // T
    loads = (csum[b[:, 1:]] - csum[b[:, :-1]]).ravel()
    return loads.max() / (total / (R * T))
```

### Balanced (nzbalanced) Partitioning

Record-balanced (work-balanced) cut: divide records into equal-load chunks.

```python
def balanced_imbalance(csum, n, R, T):
    """Nearest-boundary record-balanced cut."""
    rc = _nearest_cuts(csum, [0], [n], R)[0]  # R ranks
    tc = _nearest_cuts(csum, rc[:-1], rc[1:], T)  # T threads
    loads = (csum[tc[:, 1:]] - csum[tc[:, :-1]]).ravel()
    return loads.max() / (csum[n] / (R * T))
```

The difference between these tells you the performance ceiling—how much a better partitioner can recover.

## Parameters & Thresholds

All thresholds are **computed from this data**, not fitted:

| Parameter | Value | Source |
|-----------|-------|--------|
| `PLACEMENT_RUNG` | 0.084 (8.4%) | Smallest measured seam-doubling effect |
| `PLACEMENT_SPAN` | 0.320 (32.0%) | Grouped vs. fully alternating placement |
| `NOISE_FLOOR` | 0.0239 (2.39%) | Within-repetition spread (measurement uncertainty) |
| `p*` | ~56 | Crossover from placement to partitioner (MIMIC-IV cohort) |
| `p_atom` | 1,242 | N / max(nᵢ) – atomicity floor (exact from data) |

## Implementation

### Dependencies

- **Python 3**: numpy
- **C**: Standard library (pthreads for Pthreads version, MPI for MPI version)

### Kernels

Four isomorphic implementations of ICU feature extraction:

- `kernel_serial.c`: Single-threaded oracle
- `kernel_omp.c`: OpenMP parallelism
- `kernel_pthreads.c`: POSIX threads
- `kernel_mpi.c`: Distributed MPI

All accept the same CSR input and produce bit-identical output for validation.

### Partitioners

**C**: See `csr.py` for specification; `kernel_*.c` implement block and record-balanced variants.

**Python**: Full vectorised implementations in `pstar_curve.py` and `skewcast.py`; both implementations agree to ten decimal places.

## Validation & Reproducibility

- `validate.py`: Bit-identity checks (every run vs. serial oracle)
- `c3_bitidentity_audit.py`: Partitioner correctness audit
- `ml_validate.py`: Cross-platform consistency verification
- Experiment manifests and result logs in `results/manifest/` and `results/variants/`

## Scope

### Reported

This tool computes (arithmetic on offsets, no execution):

- Which optimisation repays attention at a given worker count
- The ceiling on what any **contiguous** partitioner can recover
- The worker count past which whole-unit assignment cannot help

### NOT Reported

(No timing model):

- Runtime, memory, node-hours, energy
- How many nodes to request to meet a deadline
- Optimisations that split units (require domain knowledge)

### Machine Independence

Thresholds (placement rungs, noise floor) are machine-independent. The **placement magnitudes** above are measured on one platform: expect the **sign** to transfer to other server-class hardware, not the magnitude.
