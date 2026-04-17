//! Inference benchmark: sweep CPU intra_threads, batch size, and DirectML.
//!
//! Usage:
//!   cargo run --release --example bench_inference -- <path/to/betaone.onnx> [iters]

use std::env;
use std::time::Instant;

use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;

const STATE_DIM: usize = 427;
const MAX_ACTIONS: usize = 30;
const ACTION_DIM: usize = 35;
const MAX_HAND: usize = 10;

fn build_session(model_path: &str, intra: i16, use_dml: bool, fixed_batch: Option<usize>) -> Result<Session, ort::Error> {
    let mut b = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra as usize)?;
    if use_dml {
        use ort::ep::{DirectML, ExecutionProvider};
        DirectML::default().register(&mut b).ok();
        if let Some(n) = fixed_batch {
            b = b.with_dimension_override("batch", n as i64)?;
        }
    }
    b.commit_from_file(model_path)
}

fn time_batch(session: &mut Session, batch: usize, iters: usize) -> f64 {
    let state = vec![0.0f32; batch * STATE_DIM];
    let acts = vec![0.0f32; batch * MAX_ACTIONS * ACTION_DIM];
    let mask = vec![false; batch * MAX_ACTIONS];
    let hand_ids = vec![2i64; batch * MAX_HAND];
    let act_ids = vec![2i64; batch * MAX_ACTIONS];

    let s = session;
    // warmup
    for _ in 0..10 {
        let inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("state".into(), Tensor::from_array(Array::from_shape_vec((batch, STATE_DIM), state.clone()).unwrap()).unwrap().into_dyn()),
            ("action_features".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_ACTIONS, ACTION_DIM), acts.clone()).unwrap()).unwrap().into_dyn()),
            ("action_mask".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_ACTIONS), mask.clone()).unwrap()).unwrap().into_dyn()),
            ("hand_card_ids".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_HAND), hand_ids.clone()).unwrap()).unwrap().into_dyn()),
            ("action_card_ids".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_ACTIONS), act_ids.clone()).unwrap()).unwrap().into_dyn()),
        ];
        let _ = s.run(inputs).unwrap();
    }

    let t = Instant::now();
    for _ in 0..iters {
        let inputs: Vec<(String, ort::value::DynValue)> = vec![
            ("state".into(), Tensor::from_array(Array::from_shape_vec((batch, STATE_DIM), state.clone()).unwrap()).unwrap().into_dyn()),
            ("action_features".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_ACTIONS, ACTION_DIM), acts.clone()).unwrap()).unwrap().into_dyn()),
            ("action_mask".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_ACTIONS), mask.clone()).unwrap()).unwrap().into_dyn()),
            ("hand_card_ids".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_HAND), hand_ids.clone()).unwrap()).unwrap().into_dyn()),
            ("action_card_ids".into(), Tensor::from_array(Array::from_shape_vec((batch, MAX_ACTIONS), act_ids.clone()).unwrap()).unwrap().into_dyn()),
        ];
        let _ = s.run(inputs).unwrap();
    }
    t.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bench_inference <path/to/betaone.onnx> [iters=2000]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let iters: usize = args.get(2).map(|s| s.parse().unwrap_or(2000)).unwrap_or(2000);

    println!("Model: {model_path}");
    println!("Iters per config: {iters}\n");
    println!("{:>30}  {:>10}  {:>14}  {:>14}  {:>12}",
             "config", "total (s)", "us/call", "us/sample", "throughput/s");

    let configs = vec![
        ("CPU intra=1, batch=1", 1, 1, false),
        ("CPU intra=2, batch=1", 2, 1, false),
        ("CPU intra=4, batch=1", 4, 1, false),
        ("CPU intra=1, batch=8", 1, 8, false),
        ("CPU intra=1, batch=32", 1, 32, false),
        ("CPU intra=1, batch=64", 1, 64, false),
        ("DML, batch=1", 1, 1, true),
        ("DML, batch=32", 1, 32, true),
        ("DML, batch=64", 1, 64, true),
        ("DML, batch=128", 1, 128, true),
    ];

    for (name, intra, batch, dml) in configs {
        let fixed_batch = if dml { Some(batch) } else { None };
        match build_session(model_path, intra, dml, fixed_batch) {
            Ok(mut sess) => {
                let t = time_batch(&mut sess, batch, iters);
                let us_call = 1e6 * t / iters as f64;
                let samples = (iters * batch) as f64;
                let us_sample = 1e6 * t / samples;
                let throughput = samples / t;
                println!("{:>30}  {:>10.3}  {:>14.1}  {:>14.1}  {:>12.0}",
                         name, t, us_call, us_sample, throughput);
            }
            Err(e) => {
                println!("{:>30}  FAILED: {e}", name);
            }
        }
    }
}
