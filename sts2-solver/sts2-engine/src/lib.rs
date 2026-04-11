use pyo3::prelude::*;

pub mod types;
pub mod effects;
pub mod combat;
pub mod cards;
pub mod actions;
pub mod enemy;
pub mod encode;
pub mod mcts;
pub mod search;
pub mod inference;
pub mod option_eval;
pub mod simulator;
pub mod ffi;

/// STS2 combat engine — Rust implementation for fast self-play.
#[pymodule]
fn sts2_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ffi::health_check, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::fight_combat, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::play_all_games, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::step, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::mcts_search, m)?)?;
    Ok(())
}
