//! Standalone CLI executable for AI Designer (`ai-designer`).

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use spodeian_telemetry::init_default;

fn main() {
    init_default();
    let args: Vec<String> = std::env::args().collect();
    match ai_designer_cli::run_cli(&args) {
        Ok(true) => std::process::exit(0),
        Ok(false) => {
            ai_designer_cli::print_help();
            std::process::exit(0);
        }
        Err(err) => {
            eprintln!("[Error] {err}");
            std::process::exit(1);
        }
    }
}
