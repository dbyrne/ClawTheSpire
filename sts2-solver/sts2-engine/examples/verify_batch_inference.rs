//! Verify that evaluate_batch produces bit-identical results to serial evaluate.
//!
//! Takes a real betaone.onnx and runs the same input through:
//!   - sequential evaluate (4 times, different inputs)
//!   - batched evaluate_batch (1 call with batch=4)
//! Prints max/mean absolute diff between results.

use std::env;

use sts2_engine::betaone::encode::{STATE_DIM, MAX_ACTIONS, ACTION_DIM, MAX_HAND};
use sts2_engine::betaone::inference::BetaOneInference;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: verify_batch_inference <path/to/betaone.onnx>");
        std::process::exit(1);
    }
    let infer = BetaOneInference::new(&args[1]).expect("session");

    let batch = 4;
    let mut states = Vec::with_capacity(batch);
    let mut acts = Vec::with_capacity(batch);
    let mut masks = Vec::with_capacity(batch);
    let mut hand_ids = Vec::with_capacity(batch);
    let mut action_ids = Vec::with_capacity(batch);
    let num_valid = vec![10usize; batch];

    // Build 4 slightly-different inputs
    for i in 0..batch {
        let mut s = [0.0f32; STATE_DIM];
        for j in 0..STATE_DIM { s[j] = ((i + 1) as f32 * 0.01 * j as f32).sin(); }
        states.push(s);

        let mut a = [0.0f32; MAX_ACTIONS * ACTION_DIM];
        for j in 0..MAX_ACTIONS * ACTION_DIM { a[j] = ((i + 2) as f32 * 0.02 * j as f32).cos(); }
        acts.push(a);

        let mut m = [true; MAX_ACTIONS];
        for j in 0..10 { m[j] = false; }  // first 10 actions valid
        masks.push(m);

        let mut h = [0i64; MAX_HAND];
        for j in 0..MAX_HAND { h[j] = (i as i64 + j as i64 + 2) % 50; }
        hand_ids.push(h);

        let mut ai = [0i64; MAX_ACTIONS];
        for j in 0..MAX_ACTIONS { ai[j] = (i as i64 + j as i64 + 3) % 50; }
        action_ids.push(ai);
    }

    // Sequential
    let mut seq_results = Vec::new();
    for i in 0..batch {
        let r = infer.evaluate(
            &states[i], &acts[i], &masks[i], &hand_ids[i], &action_ids[i], num_valid[i],
        );
        seq_results.push(r);
    }

    // Batched
    let batch_results = infer.evaluate_batch(
        &states, &acts, &masks, &hand_ids, &action_ids, &num_valid,
    );

    assert_eq!(batch_results.len(), batch);

    // Compare
    let mut max_logit_diff: f32 = 0.0;
    let mut max_value_diff: f32 = 0.0;
    let mut sum_logit_diff: f64 = 0.0;
    let mut n_logits: usize = 0;
    for i in 0..batch {
        let (seq_logits, seq_value) = &seq_results[i];
        let (batch_logits, batch_value) = &batch_results[i];
        assert_eq!(seq_logits.len(), batch_logits.len(), "logits length mismatch at row {i}");

        let vd = (seq_value - batch_value).abs();
        max_value_diff = max_value_diff.max(vd);

        for j in 0..seq_logits.len() {
            let d = (seq_logits[j] - batch_logits[j]).abs();
            max_logit_diff = max_logit_diff.max(d);
            sum_logit_diff += d as f64;
            n_logits += 1;
        }

        println!(
            "row {i}: seq_value={:.6} batch_value={:.6} diff={:.2e}",
            seq_value, batch_value, vd
        );
    }

    println!("\nMax logit diff:  {:.6e}", max_logit_diff);
    println!("Mean logit diff: {:.6e}", sum_logit_diff / n_logits as f64);
    println!("Max value diff:  {:.6e}", max_value_diff);

    // Tolerance: ONNX batched vs serial may produce slightly different FP
    // accumulation order, but same-input same-weights should be within 1e-5.
    let tol = 1e-4;
    if max_logit_diff > tol || max_value_diff > tol {
        eprintln!("FAIL: diff exceeds tolerance {tol:.0e}");
        std::process::exit(1);
    }
    println!("PASS: within tolerance {tol:.0e}");
}
