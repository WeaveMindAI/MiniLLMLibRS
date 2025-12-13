//! MiniLLMLib CLI - JSON repair tool
//!
//! Usage: `minillmlib-cli [FILE]`
//!
//! If no file is provided, reads from stdin.

use minillmlib::json_repair::{repair_json, RepairOptions};
use std::io::{self, Read};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Simple CLI - read from file or stdin
    let input = if args.len() > 1 {
        std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
            eprintln!("Error reading file: {}", e);
            std::process::exit(1);
        })
    } else {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer).unwrap_or_else(|e| {
            eprintln!("Error reading stdin: {}", e);
            std::process::exit(1);
        });
        buffer
    };

    let options = RepairOptions::default();

    match repair_json(&input, &options) {
        Ok(repaired) => println!("{}", repaired),
        Err(e) => {
            eprintln!("Error repairing JSON: {}", e);
            std::process::exit(1);
        }
    }
}
