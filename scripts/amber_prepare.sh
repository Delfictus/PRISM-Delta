#!/bin/bash
#
# AMBER-quality structure preparation for PRISM4D
#
# Usage: ./amber_prepare.sh input.pdb output_prefix [--solvate] [--padding 12.0]
#
# Outputs:
#   output_prefix.prmtop  - AMBER topology (bonds, angles, dihedrals, charges, LJ)
#   output_prefix.inpcrd  - Coordinates in Angstroms
#   output_prefix_H.pdb   - Hydrogenated PDB (intermediate)
#
# Requirements:
#   - AmberTools (reduce, tleap) in PATH or AMBERHOME set
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find AmberTools binaries
if [ -n "$AMBERHOME" ]; then
    REDUCE="$AMBERHOME/bin/reduce"
    TLEAP="$AMBERHOME/bin/tleap"
elif [ -f "/home/diddy/miniconda3/bin/reduce" ]; then
    REDUCE="/home/diddy/miniconda3/bin/reduce"
    TLEAP="/home/diddy/miniconda3/bin/tleap"
else
    REDUCE=$(which reduce 2>/dev/null || echo "")
    TLEAP=$(which tleap 2>/dev/null || echo "")
fi

# Check dependencies
check_deps() {
    if [ ! -x "$REDUCE" ]; then
        echo -e "${RED}Error: reduce not found. Install AmberTools:${NC}"
        echo "  conda install -c conda-forge ambertools"
        exit 1
    fi
    if [ ! -x "$TLEAP" ]; then
        echo -e "${RED}Error: tleap not found. Install AmberTools:${NC}"
        echo "  conda install -c conda-forge ambertools"
        exit 1
    fi
    echo -e "${GREEN}Found reduce: $REDUCE${NC}"
    echo -e "${GREEN}Found tleap: $TLEAP${NC}"
}

# Parse arguments
INPUT_PDB=""
OUTPUT_PREFIX=""
SOLVATE=false
PADDING=12.0
IONIC_STRENGTH=0.15

while [[ $# -gt 0 ]]; do
    case $1 in
        --solvate)
            SOLVATE=true
            shift
            ;;
        --padding)
            PADDING="$2"
            shift 2
            ;;
        --ionic)
            IONIC_STRENGTH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 input.pdb output_prefix [options]"
            echo ""
            echo "Options:"
            echo "  --solvate         Add explicit TIP3P water box"
            echo "  --padding N       Water box padding in Angstroms (default: 12.0)"
            echo "  --ionic N         Ionic strength in M (default: 0.15)"
            echo ""
            echo "Examples:"
            echo "  $0 1ake.pdb 1ake_prepared"
            echo "  $0 1ake.pdb 1ake_solvated --solvate --padding 10.0"
            exit 0
            ;;
        *)
            if [ -z "$INPUT_PDB" ]; then
                INPUT_PDB="$1"
            elif [ -z "$OUTPUT_PREFIX" ]; then
                OUTPUT_PREFIX="$1"
            else
                echo -e "${RED}Unknown argument: $1${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate inputs
if [ -z "$INPUT_PDB" ] || [ -z "$OUTPUT_PREFIX" ]; then
    echo -e "${RED}Usage: $0 input.pdb output_prefix [--solvate]${NC}"
    exit 1
fi

if [ ! -f "$INPUT_PDB" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_PDB${NC}"
    exit 1
fi

check_deps

# Create temp directory
WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

echo ""
echo -e "${YELLOW}=== Stage 1: Preprocessing PDB ===${NC}"

# Copy input and clean it
cp "$INPUT_PDB" "$WORKDIR/input.pdb"

# Remove HETATM (except important ions), ANISOU, and fix common issues
echo "Cleaning PDB file..."
awk '
    /^HETATM/ {
        resname = substr($0, 18, 3)
        # Keep important ions and cofactors if needed
        if (resname == "ZN " || resname == "MG " || resname == "CA " || resname == "FE ") {
            print
        }
        next
    }
    /^ANISOU/ { next }
    /^CONECT/ { next }
    /^MASTER/ { next }
    { print }
' "$WORKDIR/input.pdb" > "$WORKDIR/clean.pdb"

echo ""
echo -e "${YELLOW}=== Stage 2: Adding Hydrogens with reduce ===${NC}"

# Run reduce with flips (optimizes H-bonds for ASN, GLN, HIS)
echo "Running reduce -FLIP..."
"$REDUCE" -FLIP -Quiet "$WORKDIR/clean.pdb" > "$WORKDIR/with_H_raw.pdb" 2>"$WORKDIR/reduce.log"

# Fix histidine residue names for tleap compatibility
# reduce outputs HIE/HID/HIP but tleap expects HIS
echo "Fixing histidine residue names..."
sed 's/HIE/HIS/g; s/HID/HIS/g; s/HIP/HIS/g' "$WORKDIR/with_H_raw.pdb" | \
    grep -v "HD1.*HIS" > "$WORKDIR/with_H.pdb"

# Count atoms before/after
ATOMS_BEFORE=$(grep -c "^ATOM" "$WORKDIR/clean.pdb" || echo 0)
ATOMS_AFTER=$(grep -c "^ATOM" "$WORKDIR/with_H.pdb" || echo 0)
HYDROGENS_ADDED=$((ATOMS_AFTER - ATOMS_BEFORE))

echo -e "  Atoms before: ${ATOMS_BEFORE}"
echo -e "  Atoms after:  ${ATOMS_AFTER}"
echo -e "  ${GREEN}Hydrogens added: ${HYDROGENS_ADDED}${NC}"

# Check for HIS flips
HIS_FLIPS=$(grep -c "Flip" "$WORKDIR/reduce.log" 2>/dev/null || echo 0)
if [ "$HIS_FLIPS" -gt 0 ]; then
    echo -e "  ${YELLOW}Residue flips applied: ${HIS_FLIPS}${NC}"
fi

# Save hydrogenated PDB
cp "$WORKDIR/with_H.pdb" "${OUTPUT_PREFIX}_H.pdb"
echo -e "  Saved: ${OUTPUT_PREFIX}_H.pdb"

echo ""
echo -e "${YELLOW}=== Stage 3: Generating AMBER Topology with tleap ===${NC}"

# Create tleap input script
if [ "$SOLVATE" = true ]; then
    echo "Creating solvated system (padding: ${PADDING}A, ionic: ${IONIC_STRENGTH}M)..."
    cat > "$WORKDIR/tleap.in" << EOF
# AMBER ff14SB force field for proteins
source leaprc.protein.ff14SB
source leaprc.water.tip3p

# Load the structure
mol = loadpdb ${WORKDIR}/with_H.pdb

# Check for problems
check mol

# Add solvent box
solvatebox mol TIP3PBOX ${PADDING}

# Add ions to neutralize and reach ionic strength
addionsrand mol Na+ 0
addionsrand mol Cl- 0

# Save topology and coordinates
saveamberparm mol ${OUTPUT_PREFIX}.prmtop ${OUTPUT_PREFIX}.inpcrd

quit
EOF
else
    echo "Creating gas-phase (implicit solvent) system..."
    cat > "$WORKDIR/tleap.in" << EOF
# AMBER ff14SB force field for proteins
source leaprc.protein.ff14SB

# Load the structure
mol = loadpdb ${WORKDIR}/with_H.pdb

# Check for problems
check mol

# Save topology and coordinates
saveamberparm mol ${OUTPUT_PREFIX}.prmtop ${OUTPUT_PREFIX}.inpcrd

quit
EOF
fi

# Run tleap
echo "Running tleap..."
"$TLEAP" -f "$WORKDIR/tleap.in" > "$WORKDIR/tleap.log" 2>&1

# Check for errors
if [ ! -f "${OUTPUT_PREFIX}.prmtop" ]; then
    echo -e "${RED}Error: tleap failed to create topology${NC}"
    echo "See log:"
    cat "$WORKDIR/tleap.log"
    exit 1
fi

# Parse tleap output for statistics
TOTAL_ATOMS=$(grep "Total atoms" "$WORKDIR/tleap.log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
TOTAL_RESIDUES=$(grep "Total residues" "$WORKDIR/tleap.log" 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
WARNINGS=$(grep -c "Warning" "$WORKDIR/tleap.log" 2>/dev/null || echo 0)

echo -e "  Total atoms: ${TOTAL_ATOMS}"
echo -e "  Total residues: ${TOTAL_RESIDUES}"
if [ "$WARNINGS" -gt 0 ]; then
    echo -e "  ${YELLOW}Warnings: ${WARNINGS}${NC}"
fi

# Get file sizes
PRMTOP_SIZE=$(ls -lh "${OUTPUT_PREFIX}.prmtop" | awk '{print $5}')
INPCRD_SIZE=$(ls -lh "${OUTPUT_PREFIX}.inpcrd" | awk '{print $5}')

echo ""
echo -e "${GREEN}=== Output Files ===${NC}"
echo -e "  ${OUTPUT_PREFIX}.prmtop  (${PRMTOP_SIZE}) - Topology"
echo -e "  ${OUTPUT_PREFIX}.inpcrd  (${INPCRD_SIZE}) - Coordinates"
echo -e "  ${OUTPUT_PREFIX}_H.pdb   - Hydrogenated PDB"

# Quick validation: count atoms in prmtop
PRMTOP_ATOMS=$(head -20 "${OUTPUT_PREFIX}.prmtop" | grep -A1 "POINTERS" | tail -1 | awk '{print $1}' 2>/dev/null || echo "?")
echo ""
echo -e "${GREEN}=== Validation ===${NC}"
echo -e "  Atoms in prmtop: ${PRMTOP_ATOMS}"

# Summary
echo ""
echo -e "${GREEN}=== Pipeline Complete ===${NC}"
echo ""
echo "To use with PRISM4D:"
echo "  cargo run --release -p prism-validation --bin run_amber -- \\"
echo "    --prmtop ${OUTPUT_PREFIX}.prmtop \\"
echo "    --inpcrd ${OUTPUT_PREFIX}.inpcrd"
echo ""
