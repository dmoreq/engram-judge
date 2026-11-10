//! Binary entry point: parse once, run once, map failures to the documented exit codes.

use clap::Parser;
use engram_judge::cli::Cli;

fn main() {
    let cli = Cli::parse();
    match engram_judge::run::run(cli) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("eg-judge: {error}");
            std::process::exit(error.exit_code());
        }
    }
}
