//! PyO3 bindings: expose fight_combat() to Python.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use crate::encode::Vocabs;
use crate::types::*;

/// Parse vocabs from a Python dict of dicts.
pub fn parse_vocabs(vocab_json: &str) -> Result<Vocabs, serde_json::Error> {
    let raw: HashMap<String, HashMap<String, i64>> = serde_json::from_str(vocab_json)?;
    Ok(Vocabs {
        cards: raw.get("cards").cloned().unwrap_or_default(),
        powers: raw.get("powers").cloned().unwrap_or_default(),
        relics: raw.get("relics").cloned().unwrap_or_default(),
        intent_types: raw.get("intent_types").cloned().unwrap_or_default(),
        acts: raw.get("acts").cloned().unwrap_or_default(),
        bosses: raw.get("bosses").cloned().unwrap_or_default(),
        room_types: raw.get("room_types").cloned().unwrap_or_default(),
    })
}

/// Result of a single combat, returned to Python.
#[pyclass]
pub struct CombatResult {
    #[pyo3(get)]
    pub outcome: String,
    #[pyo3(get)]
    pub hp_after: i32,
    #[pyo3(get)]
    pub turns: i32,
    #[pyo3(get)]
    pub initial_value: f32,
}

/// Health check — verify the Rust engine is loadable.
#[pyfunction]
pub fn health_check() -> String {
    "sts2_engine OK (Rust)".to_string()
}

/// Get engine version info.
#[pyfunction]
pub fn engine_info() -> PyResult<String> {
    Ok(format!(
        "sts2_engine v{} (Rust {} + ort)",
        env!("CARGO_PKG_VERSION"),
        rustc_version(),
    ))
}

fn rustc_version() -> &'static str {
    "1.94+"
}
