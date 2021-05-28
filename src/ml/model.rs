extern crate num_traits;
extern crate ndarray;

use std::{vec::Vec, string::String, error::Error};
use crate::ml::{label::Label, error::ModelError};
use self::ndarray::{prelude::*, Array};
use self::num_traits::ToPrimitive;
use core::str::FromStr;

pub trait Model {    
    fn from_labels(file_path: &String) -> Self;

    fn from_json(file_path: &String) -> Self;

    fn to_json(&self, file_path: &String);

    fn train<Num: ToPrimitive + Copy + FromStr>(&mut self, file_path: &String) 
        -> Result<(), Box<dyn Error>>;

    fn test<Num: ToPrimitive + Copy + FromStr>
        (&self, file_path: &String, multithreaded: bool) 
        -> Result<f64, Box<dyn Error>>;

    fn classify<Num: ToPrimitive + Copy + FromStr>
        (&self, sample_features: &Vec<String>) 
        -> Result<Box<dyn Label>, ModelError>;

    fn calculate_accuracy(confusion_matrix: &Array<usize, Ix2>) -> f64 {
        confusion_matrix.diag().sum() as f64 / confusion_matrix.sum() as f64
    }
} 
