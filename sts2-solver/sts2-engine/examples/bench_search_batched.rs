//! Benchmark sequential vs batched MCTS search on a real BetaOne model.
//!
//! Usage:
//!   cargo run --release --example bench_search_batched -- <path/to/betaone.onnx> <path/to/card_vocab.json>

use std::env;
use std::fs;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use sts2_engine::betaone::encode::CardVocab;
use sts2_engine::betaone::inference::BetaOneInference;
use sts2_engine::betaone::mcts_adapter::BetaOneMCTSAdapter;
use sts2_engine::mcts::MCTS;
use sts2_engine::types::*;

fn build_test_state() -> CombatState {
    // Minimal realistic combat state: Silent deck vs a single enemy.
    let strike = Card {
        id: "STRIKE_SILENT".into(),
        name: "Strike".into(),
        cost: 1,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(6),
        tags: ["Strike".into()].into(),
        ..Default::default()
    };
    let defend = Card {
        id: "DEFEND_SILENT".into(),
        name: "Defend".into(),
        cost: 1,
        card_type: CardType::Skill,
        target: TargetType::Self_,
        block: Some(5),
        ..Default::default()
    };
    let neutralize = Card {
        id: "NEUTRALIZE".into(),
        name: "Neutralize".into(),
        cost: 0,
        card_type: CardType::Attack,
        target: TargetType::AnyEnemy,
        damage: Some(3),
        powers_applied: vec![("Weak".into(), 1)],
        ..Default::default()
    };
    let survivor = Card {
        id: "SURVIVOR".into(),
        name: "Survivor".into(),
        cost: 1,
        card_type: CardType::Skill,
        target: TargetType::Self_,
        block: Some(8),
        ..Default::default()
    };

    let hand = vec![
        strike.clone(),
        strike.clone(),
        defend.clone(),
        defend.clone(),
        neutralize.clone(),
        survivor.clone(),
    ];
    let draw_pile = vec![strike; 4];

    let enemy = EnemyState {
        id: "TEST_ENEMY".into(),
        name: "Test".into(),
        hp: 45,
        max_hp: 45,
        intent_type: Some("Attack".into()),
        intent_damage: Some(12),
        intent_hits: 1,
        ..Default::default()
    };

    CombatState {
        player: PlayerState {
            hp: 55,
            max_hp: 70,
            energy: 3,
            max_energy: 3,
            hand,
            draw_pile,
            ..Default::default()
        },
        enemies: vec![enemy],
        ..Default::default()
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bench_search_batched <betaone.onnx> <card_vocab.json> [sims=1000] [runs=5]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let vocab_path = &args[2];
    let sims: usize = args.get(3).map(|s| s.parse().unwrap_or(1000)).unwrap_or(1000);
    let runs: usize = args.get(4).map(|s| s.parse().unwrap_or(5)).unwrap_or(5);

    let vocab_json = fs::read_to_string(vocab_path).expect("read vocab");
    let vocab: CardVocab = serde_json::from_str(&vocab_json).expect("parse vocab");
    let inference = BetaOneInference::new(model_path).expect("load ONNX");
    let adapter = BetaOneMCTSAdapter::new(&inference, &vocab);

    let state = build_test_state();
    let card_db = CardDB::default();

    let mut mcts = MCTS::new(&card_db, &adapter);
    mcts.turn_boundary_eval = true;
    mcts.add_noise = false;

    println!("sims per search: {sims}");
    println!("runs per config: {runs}\n");

    let configs: Vec<(&str, Option<usize>)> = vec![
        ("sequential",       None),
        ("batched K=4",   Some(4)),
        ("batched K=8",   Some(8)),
        ("batched K=16",  Some(16)),
        ("batched K=32",  Some(32)),
        ("batched K=64",  Some(64)),
    ];

    println!("{:>18}  {:>10}  {:>10}  {:>8}", "config", "mean (ms)", "min (ms)", "speedup");
    println!("{}", "-".repeat(52));

    let mut baseline_mean = 0.0f64;
    for (name, batch) in configs {
        let mut times = Vec::with_capacity(runs);
        for r in 0..runs {
            let mut rng = StdRng::seed_from_u64(42 + r as u64);
            let t = Instant::now();
            let _ = match batch {
                None => mcts.search(&state, sims, 1.0, &mut rng),
                Some(k) => mcts.search_batched(&state, None, sims, k, 1.0, &mut rng),
            };
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        if batch.is_none() { baseline_mean = mean; }
        let speedup = if mean > 0.0 { baseline_mean / mean } else { 0.0 };
        println!("{:>18}  {:>10.1}  {:>10.1}  {:>7.2}x", name, mean, min, speedup);
    }
}
