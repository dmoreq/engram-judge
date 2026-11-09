//! Binary entry point: parse once, run once, map failures to the documented exit codes.

use clap::Parser;
use engram_laya::cli::Cli;

fn main() {
    let cli = Cli::parse();
    match engram_laya::run::run(cli) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("eg-laya: {error}");
            std::process::exit(error.exit_code());
        }
    }
}
