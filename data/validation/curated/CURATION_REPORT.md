# PRISM-4D Data Curation Report

**Generated**: 2026-01-09 04:36:49.817308135 UTC

**Manifest BLAKE3**: `c075332825eda2c1ee5ffecfe8af394c5ea387d89d81dd1803e043c8c1707ee9`

## Executive Summary

This document provides cryptographic provenance for 6 validation targets used in PRISM-4D retrospective blind validation. 6 targets (100%) meet temporal integrity requirements for scientifically defensible blind testing.

## Data Leakage Prevention

For retrospective blind validation to be scientifically defensible:

1. **APO structures must predate drug discovery** - We only use the APO structure 
   (closed/inactive state) as input, and verify it was deposited before the drug was discovered.
2. **HOLO structures are ground truth only** - The HOLO structure (open/drug-bound state) 
   is NEVER used during simulation, only for evaluation after the fact.
3. **No binding site information encoded** - The pocket residues are not provided to 
   PRISM-NOVA during simulation; they're only used to evaluate if the correct site was found.

## Therapeutic Area Coverage

| Area | Targets |
|------|--------:|
| Metabolic | 1 |
| Infectious | 2 |
| Oncology | 3 |

## Target Details

### KRAS_G12C (Oncology)

**Drug**: Sotorasib (discovered ~2013-01-01)

#### APO Structure (Blind Input)

- **PDB ID**: 3GFT
- **BLAKE3**: `b610a9cd12b892b98303c9e573fdfceff1e2e953d7b6aa4201ceab631c7ab44e`
- **Deposition Date**: 2009-02-27
- **Atoms**: 8086
- **Residues**: 1122
- **Temporal Validity**: ✓ VALID (1404 days before drug discovery)

#### HOLO Structure (Ground Truth - Evaluation Only)

- **PDB ID**: 6OIM
- **BLAKE3**: `36b4fdcaa74cb849f4c502ad5d98f0619553d60cb9d23c51d7723a2605496534`
- **Atoms**: 1613
- **Residues**: 377

#### Notes

- Switch-II pocket

---

### BTK (Oncology)

**Drug**: Ibrutinib (discovered ~2007-01-01)

#### APO Structure (Blind Input)

- **PDB ID**: 1K2P
- **BLAKE3**: `e3a3d1769d0544776066a083aa3e22acad21a37703a1f055aa2cdd76a0678ef4`
- **Deposition Date**: 2001-09-28
- **Atoms**: 4178
- **Residues**: 516
- **Temporal Validity**: ✓ VALID (1921 days before drug discovery)

#### HOLO Structure (Ground Truth - Evaluation Only)

- **PDB ID**: 5P9J
- **BLAKE3**: `0c44259f5ced839450807cc6b7f11f75af21410269c9186f1b7bb72bdc5bdcd1`
- **Atoms**: 2369
- **Residues**: 439

#### Notes

- C481 covalent site

---

### BRAF_V600E (Oncology)

**Drug**: Vemurafenib (discovered ~2006-01-01)

#### APO Structure (Blind Input)

- **PDB ID**: 1UWH
- **BLAKE3**: `d0520857b31b0194e0ee00dd2a9289bb478cfe121c2dddfc75dea8b86a9cc194`
- **Deposition Date**: 2004-02-05
- **Atoms**: 4290
- **Residues**: 564
- **Temporal Validity**: ✓ VALID (696 days before drug discovery)

#### HOLO Structure (Ground Truth - Evaluation Only)

- **PDB ID**: 3OG7
- **BLAKE3**: `06b028aa6bce056b3ae2f64eb6da3121ca8fb6d1a4dab5ab3d619456dc174de3`
- **Atoms**: 4100
- **Residues**: 567

#### Notes

- DFG-out/αC-helix-out pocket

---

### PTP1B (Metabolic)

**Drug**: Trodusquemine (discovered ~2010-01-01)

#### APO Structure (Blind Input)

- **PDB ID**: 2HNP
- **BLAKE3**: `9d10de6cd47ed4027bf403cc407fd0ed801c00dbcfc065c0feebac742e5c3978`
- **Deposition Date**: 1994-09-19
- **Atoms**: 2270
- **Residues**: 278
- **Temporal Validity**: ✓ VALID (5583 days before drug discovery)

#### HOLO Structure (Ground Truth - Evaluation Only)

- **PDB ID**: 1T49
- **BLAKE3**: `4c8f4f8071ffb3d1311eb733392245eb58ce6bcab9a003471b68c6e4aa376207`
- **Atoms**: 2577
- **Residues**: 519

#### Notes

- Allosteric C-terminal site

---

### HIV_RT (Infectious)

**Drug**: Rilpivirine (discovered ~2004-01-01)

#### APO Structure (Blind Input)

- **PDB ID**: 1DLO
- **BLAKE3**: `66e6e3ec545e9e5cee851ac6c206856544699287158841e45f4823e32973d063`
- **Deposition Date**: 1996-04-17
- **Atoms**: 7691
- **Residues**: 971
- **Temporal Validity**: ✓ VALID (2815 days before drug discovery)

#### HOLO Structure (Ground Truth - Evaluation Only)

- **PDB ID**: 4G1Q
- **BLAKE3**: `16b4f16fc7247a5b494031032212b77ab8256614310db621bb2de8111b3e3e3e`
- **Atoms**: 16771
- **Residues**: 1809

#### Notes

- NNRTI allosteric pocket

---

### HCV_NS3 (Infectious)

**Drug**: Glecaprevir (discovered ~2012-01-01)

#### APO Structure (Blind Input)

- **PDB ID**: 1A1R
- **BLAKE3**: `5275a722571d35c0243774c1fd487bbb23ec72e04d29e7ccaf892c3f74be79a5`
- **Deposition Date**: 1997-12-15
- **Atoms**: 3686
- **Residues**: 500
- **Temporal Validity**: ✓ VALID (5130 days before drug discovery)

#### HOLO Structure (Ground Truth - Evaluation Only)

- **PDB ID**: 4NWL
- **BLAKE3**: `954a74f0e2c4fc63549fd13b7c7d02c0ced56176193e3cfb30ac2172c6b4608c`
- **Atoms**: 3092
- **Residues**: 504

#### Notes

- NS3/4A protease site

---

## Provenance Verification

To verify the integrity of this dataset:

```bash
# Verify manifest hash
blake3sum curation_manifest.json

# Verify individual PDB files
for pdb in pdb/apo/*.pdb pdb/holo/*.pdb; do
    echo "$pdb: $(blake3sum $pdb | cut -d' ' -f1)"
done
```

## Legal & Ethical Statement

All PDB structures are obtained from the RCSB Protein Data Bank, a publicly 
available resource. The temporal validation ensures that our retrospective 
blind testing is scientifically valid and does not constitute "p-hacking" 
or cherry-picking of favorable results.
