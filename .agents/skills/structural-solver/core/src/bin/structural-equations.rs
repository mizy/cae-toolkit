#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: structural-equations case.json [auto|gpu|cpu]");
        std::process::exit(2);
    }
    let result = (|| -> Result<(), String> {
        let json = std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?;
        let a = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        let backend = args.get(2).map(String::as_str).unwrap_or("auto");
        let result = pollster::block_on(structural_core::solve(&a, backend, 1e-5, 6000))?;
        println!(
            "{}",
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        );
        if !result.converged {
            return Err("Solver did not converge".into());
        }
        Ok(())
    })();
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
#[cfg(target_arch = "wasm32")]
fn main() {}
