use rurel::mdp::State;

#[derive(PartialEq, Eq, Hash, Clone)]
struct MyState { x: i32 }
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct MyAction { dx: i32 }

impl State for MyState {
	type A = MyAction;
	fn reward(&self) -> f64 {
		// Negative Euclidean distance
        if self.x == 7 { -100. }
        else { 0. }
	}
	fn actions(&self) -> Vec<MyAction> {
		vec![MyAction { dx: 1},
		     MyAction { dx: 2},
		     MyAction { dx: 3},
		]
	}
}

use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rurel::mdp::Agent;

struct MyAgent { state: MyState, between: Uniform<i32>, rng: ThreadRng }
impl MyAgent {
    fn reset(&mut self) {
        self.state.x = 0;
    }
}
impl Agent<MyState> for MyAgent {
	fn current_state(&self) -> &MyState {
		&self.state
	}
	fn take_action(&mut self, action: &MyAction) -> () {
        let mut new_x = self.state.x + action.dx + self.between.sample(&mut self.rng);
        if new_x > 7 { new_x = 7 }
        self.state.x = new_x;
	}
}

use std::collections::HashMap;

use rurel::AgentTrainer;
use rurel::strategy::learn::QLearning;
use rurel::strategy::explore::{ ExplorationStrategy, RandomExploration };
use rurel::strategy::terminate::{ TerminationStrategy, FixedIterations };

fn main() {
    println!("Rurel test");
    let mut trainer = AgentTrainer::new();
    let mut agent = MyAgent {
        state: MyState { x: 0 },
        between: Uniform::from(1..=3),
        rng: rand::thread_rng()
    };

    // Train the agent
    //      with Q learning,
    //      with learning rate 0.2,
    //      discount factor 0.01 
    //      and an initial value of Q of 2.0.
    // Let the trainer run for 100000 iterations,
    //      randomly exploring new states.
    for epoch in 0..100_000 {
        trainer.train(&mut agent,
                      &QLearning::new(0.2, 0.01, 0.),
                      //&mut Term::new(),
                      &mut FixedIterations::new(10),
                      &RandomExploration::new());
        agent.reset();
    }
    // Query the learned value (Q) for a certain action in a certain state
    for i in 0..=7 {
        /* let entry: &HashMap<MyAction, f64> = */match trainer.expected_values(&MyState {
                x: i,
            }) {
            Some(entry) => println!("{} {:.0?}", i, entry),
            None => println!("{} Oops", i),
        };
        //let val: f64 = entry.values().sum();
        //print!("{:.0}\t", val);
        //print!("{:.0?}\t", entry.values());
        //println!("{:.0?}", entry); 
    }
}
