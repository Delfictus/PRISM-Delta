//! Test AMBER prmtop/inpcrd parser
//!
//! Usage: cargo run --bin test_prmtop -- <prmtop> <inpcrd>

use anyhow::Result;
use prism_physics::{AmberSystem, AmberPrmtop, AmberInpcrd};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <prmtop> <inpcrd>", args[0]);
        std::process::exit(1);
    }

    let prmtop_path = &args[1];
    let inpcrd_path = &args[2];

    println!("Loading AMBER system...");
    println!("  prmtop: {}", prmtop_path);
    println!("  inpcrd: {}", inpcrd_path);
    println!();

    // Load and parse
    let prmtop = AmberPrmtop::from_file(prmtop_path)?;
    let inpcrd = AmberInpcrd::from_file(inpcrd_path)?;
    let system = AmberSystem::from_files(prmtop_path, inpcrd_path)?;

    // Print summary
    println!("=== Topology Summary ===");
    println!("Title: {}", prmtop.title);
    println!("Atoms: {}", prmtop.n_atoms());
    println!("Residues: {}", prmtop.n_residues());
    println!("Atom types: {}", prmtop.pointers.ntypes);
    println!("Periodic: {}", prmtop.is_periodic());
    println!();

    // Bonds
    let bonds = prmtop.get_bonds();
    println!("=== Bonded Terms ===");
    println!("Bonds: {}", bonds.len());
    if !bonds.is_empty() {
        let (i, j, k, r0) = bonds[0];
        println!("  Example: atoms {}-{}, k={:.2} kcal/mol/A^2, r0={:.3} A", i, j, k, r0);
    }

    // Angles
    let angles = prmtop.get_angles();
    println!("Angles: {}", angles.len());
    if !angles.is_empty() {
        let (i, j, k, force_k, theta0) = angles[0];
        println!("  Example: atoms {}-{}-{}, k={:.2} kcal/mol/rad^2, theta0={:.3} rad ({:.1} deg)",
                 i, j, k, force_k, theta0, theta0.to_degrees());
    }

    // Dihedrals
    let dihedrals = prmtop.get_dihedrals();
    println!("Dihedrals: {}", dihedrals.len());
    if !dihedrals.is_empty() {
        let (i, j, k, l, force_k, n, phase, improper) = dihedrals[0];
        println!("  Example: atoms {}-{}-{}-{}, k={:.2}, n={:.0}, phase={:.3}, improper={}",
                 i, j, k, l, force_k, n, phase, improper);
    }
    println!();

    // Coordinates
    println!("=== Coordinates ===");
    println!("Atoms in inpcrd: {}", inpcrd.n_atoms);
    println!("Has velocities: {}", inpcrd.has_velocities());
    println!("Has box: {}", inpcrd.is_periodic());

    let positions = inpcrd.positions();
    if !positions.is_empty() {
        println!("  First atom: [{:.3}, {:.3}, {:.3}]",
                 positions[0][0], positions[0][1], positions[0][2]);
        let last = positions.last().unwrap();
        println!("  Last atom: [{:.3}, {:.3}, {:.3}]", last[0], last[1], last[2]);
    }
    println!();

    // Charges
    println!("=== Partial Charges ===");
    let total_charge: f64 = prmtop.charges.iter().sum();
    println!("Total charge: {:.3} e", total_charge);
    println!("  First 5 charges: {:?}", &prmtop.charges[..5.min(prmtop.charges.len())]);
    println!();

    // LJ parameters
    println!("=== Lennard-Jones ===");
    let lj_params = prmtop.get_atom_lj_params();
    if !lj_params.is_empty() {
        println!("  First atom: sigma={:.3} A, epsilon={:.4} kcal/mol",
                 lj_params[0].0, lj_params[0].1);
    }
    println!();

    // GPU arrays
    println!("=== GPU Arrays ===");
    let gpu = system.to_gpu_arrays();
    println!("{}", gpu.summary());
    println!("  Position array: {} floats", gpu.positions.len());
    println!("  Bond arrays: {} bonds", gpu.n_bonds);
    println!("  Angle arrays: {} angles", gpu.n_angles);
    println!("  Dihedral arrays: {} dihedrals", gpu.n_dihedrals);
    println!();

    // Memory estimate
    let mem_mb = (gpu.positions.len() * 4
        + gpu.masses.len() * 4
        + gpu.charges.len() * 4
        + gpu.lj_sigma.len() * 4
        + gpu.lj_epsilon.len() * 4
        + gpu.bond_i.len() * 4 * 4  // i, j, k, r0
        + gpu.angle_i.len() * 5 * 4 // i, j, k, force_k, theta0
        + gpu.dihedral_i.len() * 7 * 4) as f64 / 1024.0 / 1024.0;
    println!("Estimated GPU memory: {:.2} MB", mem_mb);

    println!("\nAMBER system loaded successfully!");
    Ok(())
}
