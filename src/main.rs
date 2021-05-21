#[macro_use]
extern crate approx;
extern crate csv;

mod naivebayes;

use naivebayes::model::GaussianNaiveBayes;
use std::string::String;

fn main() {
    let label_path: String = String::from("./data/emnist-balanced-mapping.txt");
    let train_path: String = String::from("./data/emnist-balanced-train.csv");
    let test_path: String = String::from("./data/emnist-balanced-test.csv");

    let mut model: &mut GaussianNaiveBayes = &mut GaussianNaiveBayes::from_labels(&label_path);

    model.train::<u8>(&train_path);
}