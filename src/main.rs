#[macro_use]
extern crate approx;
extern crate csv;

mod naivebayes;
mod ml;

use naivebayes::model::GaussianNaiveBayes;
use clap::{App, load_yaml};
use std::string::String;
use ml::model::Model;

fn run_app() -> Result<(), String> {
    let yaml = load_yaml!("cli-arguments.yaml");
    let arg_matches = App::from(yaml).get_matches();

    // TODO verbosity, confusion

    let model: Box<GaussianNaiveBayes> = 
            match (arg_matches.value_of("labels"), arg_matches.value_of("load")) {
        (None, None) => return Err(String::from("No model loading method provided.")),
        (Some(_), Some(_)) => return Err(String::from("Redundant model loading methods provided. Choose only one!")),
        (Some(path), None) => {
            let label_path: String = String::from(path);

            // Train model here since label path was provided
            let mut untrained: GaussianNaiveBayes = 
                GaussianNaiveBayes::from_labels(&label_path);

            match arg_matches.value_of("train") {
                Some(tp) => {
                    let train_path: String = String::from(tp);

                    match untrained.train::<u8>(&train_path) {
                        Ok(_) => println!("Model trained."),
                        Err(_) => return Err(String::from("Model training failed"))
                    };
                }
                None => return Err(String::from("Label training path provided but no training data given."))
            }
            Box::new(untrained)
        },
        (None, Some(path)) => {
            let model_path: String = String::from(path);
            Box::new(GaussianNaiveBayes::from_json(&model_path))
        }
    };

    if let Some(save_path) = arg_matches.value_of("save") {
        model.to_json(&String::from(save_path));
    }

    if let Some(test_path) = arg_matches.value_of("test") {
        let path: String = String::from(test_path);
        match model.test::<u8>(&path) {
            Ok(accuracy) => println!("Model accuracy: {}", accuracy),
            Err(_) => return Err(String::from("Model testing failed"))
        };
    }

    Ok(())
}

fn main() {
    if let Err(error_message) = run_app() {
        println!("{}", error_message);
    }
}
