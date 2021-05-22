#[allow(dead_code)]
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

    // println!("{:?}", arg_matches);

    let labels_path: Option<String> = match arg_matches.value_of("labels") {
        Some(label_path) => Some(String::from(label_path)),
        None => None
    };

    let model_json_path: Option<String> = match arg_matches.value_of("load") {
        Some(model_path) => Some(String::from(model_path)),
        None => None
    };

    let model: Box<GaussianNaiveBayes> = match (labels_path, model_json_path) {
        (None, None) => return Err(String::from("No model loading method provided.")),
        (Some(_), Some(_)) => return Err(String::from("Redundant model loading methods provided. Choose only one!")),
        (Some(path), None) => {
            // Train model here since label path was provided
            let mut untrained: GaussianNaiveBayes = GaussianNaiveBayes::from_labels(&path);
            match arg_matches.value_of("train") {
                Some(train_path) => {
                    let path: String = String::from(train_path);

                    match untrained.train::<u8>(&path) {
                        Ok(_) => println!("Model trained."),
                        Err(_) => return Err(String::from("Model training failed"))
                    };
                }
                None => return Err(String::from("Label training path provided but no training data given."))
            }
            Box::new(untrained)
        },
        (None, Some(path)) => Box::new(GaussianNaiveBayes::from_json(&path))
    };

    match arg_matches.value_of("save") {
        Some(save_path) => {
            let path: String = String::from(save_path);
            model.to_json(&path);
        }
        None => ()
    }; 

    match arg_matches.value_of("test") {
        Some(test_path) => {
            let path: String = String::from(test_path);
            match model.test::<u8>(&path) {
                Ok(accuracy) => println!("Model accuracy: {}", accuracy),
                Err(_) => return Err(String::from("Model testing failed"))
            };
        }
        None => ()
    }; 

    Ok(())

    // let label_path: String = String::from("./data/digits/emnist-digits-mapping.txt");
    // let train_path: String = String::from("./data/digits/emnist-digits-train.csv");
    // let test_path: String = String::from("./data/digits/emnist-digits-test.csv");

    // let label_path: String = String::from("./data/balanced/emnist-balanced-mapping.txt");
    // let train_path: String = String::from("./data/balanced/emnist-balanced-train.csv");
    // let test_path: String = String::from("./data/balanced/emnist-balanced-test.csv");

    // let balanced_path: String = String::from("./models/balanced/balanced.json");

    // let model: &mut GaussianNaiveBayes = &mut GaussianNaiveBayes::from_labels(&label_path);
    // let model: &mut GaussianNaiveBayes = &mut GaussianNaiveBayes::from_json(&balanced_path);

    // match model.train::<u8>(&train_path) {
    //     Ok(_) => println!("Model trained."),
    //     Err(_) => println!("Model training failed.")
    // };

    // match model.test::<u8>(&test_path) {
    //     Ok(accuracy) => println!("Model accuracy: {}", accuracy),
    //     Err(_) => println!("Model testing failed.")
    // };

    // model.to_json(&String::from("./models/balanced/balanced.json"));
}

fn main() {
    match run_app() {
        Ok(_) => (),
        Err(error_message) => println!("{}", error_message)
    };
}
