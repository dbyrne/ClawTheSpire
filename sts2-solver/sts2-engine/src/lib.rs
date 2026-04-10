use pyo3::prelude::*;

pub mod types;
pub mod effects;
pub mod combat;
pub mod cards;
pub mod actions;
pub mod enemy;
pub mod mcts;

/// STS2 combat engine — Rust implementation for fast self-play.
#[pymodule]
fn sts2_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(health_check, m)?)?;
    Ok(())
}

#[pyfunction]
fn health_check() -> String {
    "sts2_engine OK".to_string()
}
