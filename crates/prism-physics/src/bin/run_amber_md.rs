//! Run GPU MD from AMBER prmtop/inpcrd files
//!
//! This demonstrates the full pipeline:
//! 1. Load AMBER-quality topology from tleap
//! 2. Initialize GPU engine with validated parameters
//! 3. Run short MD simulation
//! 4. Report energy and structure quality
//!
//! Usage:
//!   cargo run --release -p prism-physics --features cuda --bin run_amber_md -- \
//!     --prmtop system.prmtop --inpcrd system.inpcrd \
//!     [--steps 1000] [--dt 0.001] [--temp 300]

use anyhow::{Context, Result};
use clap::Parser;
use cudarc::driver::CudaContext;
use prism_gpu::AmberMegaFusedHmc;
use prism_physics::AmberSystem;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "run_amber_md")]
#[command(about = "Run GPU MD simulation from AMBER topology files")]
struct Args {
    /// AMBER topology file (prmtop)
    #[arg(long)]
    prmtop: String,

    /// AMBER coordinate file (inpcrd)
    #[arg(long)]
    inpcrd: String,

    /// Number of MD steps
    #[arg(long, default_value = "1000")]
    steps: usize,

    /// Time step in picoseconds
    #[arg(long, default_value = "0.001")]
    dt: f64,

    /// Temperature in Kelvin
    #[arg(long, default_value = "300.0")]
    temp: f64,

    /// Langevin friction coefficient in fs^-1
    #[arg(long, default_value = "1.0")]
    gamma: f64,

    /// Enable minimization before MD
    #[arg(long)]
    minimize: bool,

    /// Number of minimization steps
    #[arg(long, default_value = "100")]
    min_steps: usize,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       PRISM4D GPU MD Engine - AMBER Topology Loader           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Load AMBER system
    println!("Loading AMBER system...");
    println!("   prmtop: {}", args.prmtop);
    println!("   inpcrd: {}", args.inpcrd);

    let load_start = Instant::now();
    let system = AmberSystem::from_files(&args.prmtop, &args.inpcrd)
        .context("Failed to load AMBER system")?;
    println!("   Loaded in {:.2?}", load_start.elapsed());
    println!();

    // Print system info
    println!("System Summary:");
    println!("   Atoms: {}", system.n_atoms());
    println!("   Residues: {}", system.prmtop.n_residues());
    println!("   Periodic: {}", system.is_periodic());
    if let Some(box_dims) = system.box_dimensions() {
        println!("   Box: {:.2} x {:.2} x {:.2} A", box_dims[0], box_dims[1], box_dims[2]);
    }

    // Get topology data for GPU
    let topo = system.to_gpu_topology();
    println!("   Bonds: {}", topo.bonds.len());
    println!("   Angles: {}", topo.angles.len());
    println!("   Dihedrals: {}", topo.dihedrals.len());

    // Calculate total charge
    let total_charge: f32 = topo.nb_params.iter().map(|(_, _, c, _)| c).sum();
    println!("   Total charge: {:.2} e", total_charge);
    println!();

    // Initialize CUDA
    println!("Initializing CUDA...");
    let cuda_start = Instant::now();
    let context = CudaContext::new(0).context("Failed to create CUDA context")?;
    println!("   CUDA init: {:.2?}", cuda_start.elapsed());
    println!();

    // Create MD engine
    println!("Creating MD engine...");
    let engine_start = Instant::now();
    // Note: CudaContext::new already returns Arc<CudaContext>
    let mut engine = AmberMegaFusedHmc::new(context, system.n_atoms())
        .context("Failed to create MD engine")?;
    println!("   Engine created in {:.2?}", engine_start.elapsed());

    // Upload topology
    println!("Uploading topology to GPU...");
    let upload_start = Instant::now();
    engine.upload_topology(
        &topo.positions,
        &topo.bonds,
        &topo.angles,
        &topo.dihedrals,
        &topo.nb_params,
        &topo.exclusions,
    ).context("Failed to upload topology")?;
    println!("   Upload complete in {:.2?}", upload_start.elapsed());
    println!();

    // Optional minimization
    if args.minimize {
        println!("Running energy minimization ({} steps)...", args.min_steps);
        let min_start = Instant::now();
        let final_energy = engine.minimize(args.min_steps, 0.001)
            .context("Minimization failed")?;
        println!("   Final PE: {:.2} kcal/mol", final_energy);
        println!("   Minimization time: {:.2?}", min_start.elapsed());
        println!();
    }

    // Run MD
    println!("Running MD simulation...");
    println!("   Steps: {}", args.steps);
    println!("   dt: {} ps", args.dt);
    println!("   Temperature: {} K", args.temp);
    println!("   Gamma: {} fs^-1", args.gamma);
    println!();

    let md_start = Instant::now();
    let result = engine.run(
        args.steps,
        args.dt as f32,
        args.temp as f32,
        args.gamma as f32,
    ).context("MD simulation failed")?;
    let md_time = md_start.elapsed();

    // Results
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                     SIMULATION RESULTS                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Performance:");
    println!("   Total time: {:.2?}", md_time);
    println!("   Steps/sec: {:.1}", args.steps as f64 / md_time.as_secs_f64());
    println!("   ns/day: {:.2}", args.steps as f64 * args.dt * 1e-3 / md_time.as_secs_f64() * 86400.0);
    println!();

    println!("Energy:");
    println!("   Potential: {:.2} kcal/mol", result.potential_energy);
    println!("   Kinetic: {:.2} kcal/mol", result.kinetic_energy);
    println!("   Total: {:.2} kcal/mol", result.potential_energy + result.kinetic_energy);
    println!();

    println!("Temperature:");
    println!("   Average: {:.1} K (target: {:.1} K)", result.avg_temperature, args.temp);
    println!("   DOF: {} (constraints: {} SETTLE, {} H-bond)",
             result.n_dof,
             result.constraint_info.n_settle_constraints,
             result.constraint_info.n_h_constraints);
    println!();

    // Energy trajectory
    if !result.energy_trajectory.is_empty() {
        let first = &result.energy_trajectory[0];
        let last = result.energy_trajectory.last().unwrap();
        let energy_drift = last.total_energy - first.total_energy;
        let drift_per_ns = energy_drift / (args.steps as f64 * args.dt);

        println!("Energy Stability:");
        println!("   Initial E: {:.2} kcal/mol", first.total_energy);
        println!("   Final E: {:.2} kcal/mol", last.total_energy);
        println!("   Drift: {:.4} kcal/mol/ns", drift_per_ns);

        // Warn if drift is too high
        if drift_per_ns.abs() > 10.0 {
            println!("   Warning: Energy drift is high! Consider smaller timestep.");
        }
    }
    println!();

    // Structure quality check (RMSD from initial)
    let initial_positions = &topo.positions;
    let final_positions = &result.positions;

    if initial_positions.len() == final_positions.len() {
        let mut rmsd_sum = 0.0;
        let n = initial_positions.len() / 3;
        for i in 0..n {
            let dx = final_positions[i*3] - initial_positions[i*3];
            let dy = final_positions[i*3+1] - initial_positions[i*3+1];
            let dz = final_positions[i*3+2] - initial_positions[i*3+2];
            rmsd_sum += dx*dx + dy*dy + dz*dz;
        }
        let rmsd = (rmsd_sum / n as f32).sqrt();
        println!("Structure:");
        println!("   RMSD from initial: {:.3} A", rmsd);
    }

    println!();
    println!("Simulation complete!");

    Ok(())
}
