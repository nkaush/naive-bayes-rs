#[macro_use]
extern crate approx;
extern crate csv;

mod naivebayes;

use naivebayes::model::GaussianNaiveBayes;
use std::string::String;

fn main() {
    let label_path: String = String::from("./data/digits/emnist-digits-mapping.txt");
    let train_path: String = String::from("./data/digits/emnist-digits-train.csv");
    let test_path: String = String::from("./data/digits/emnist-digits-test.csv");

    // let label_path: String = String::from("./data/balanced/emnist-balanced-mapping.txt");
    // let train_path: String = String::from("./data/balanced/emnist-balanced-train.csv");
    // let test_path: String = String::from("./data/balanced/emnist-balanced-test.csv");

    let mut model: &mut GaussianNaiveBayes = &mut GaussianNaiveBayes::from_labels(&label_path);

    match model.train::<u8>(&train_path) {
        Ok(_) => println!("Model trained."),
        Err(_) => println!("Model training failed.")
    };

    match model.test::<u8>(&test_path) {
        Ok(accuracy) => println!("Model accuracy: {}", accuracy),
        Err(_) => println!("Model testing failed.")
    };
}