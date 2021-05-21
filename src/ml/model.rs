extern crate num_traits;

use std::{vec::Vec, string::String, error::Error};
use self::num_traits::ToPrimitive;
use crate::ml::label::Label;
use core::str::FromStr;

pub trait Model {    
    fn from_labels(file_path: &String) -> Self;

    fn from_json(file_path: &String) -> Self;

    fn to_json(&self, file_path: &String);

    fn train<Num: ToPrimitive + Copy + FromStr>(&mut self, file_path: &String) 
        -> Result<(), Box<dyn Error>>;

    fn test<Num: ToPrimitive + Copy + FromStr>(&self, file_path: &String) 
        -> Result<f64, Box<dyn Error>>;

    fn classify<Num: ToPrimitive + Copy>(&self, sample_features: &Vec<Num>) 
        -> Box<dyn Label>;
} 