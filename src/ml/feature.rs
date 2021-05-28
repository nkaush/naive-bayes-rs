extern crate num_traits;

use crate::ml::{label::Label, error::ModelError};
use self::num_traits::ToPrimitive;
use std::str::FromStr;

pub trait Feature {
    fn train_iter<Num: ToPrimitive + Copy + FromStr>
        (&mut self, label: &dyn Label, value: &String, iter: usize);

    fn prepare(&mut self);

    fn is_trained(&self) -> bool;

    fn likelihood_given_class<Num: ToPrimitive + Copy + FromStr>
        (&self, feature: &String, class: &dyn Label) -> Result<f64, ModelError>;

    fn class_likelihood(&self, class: &dyn Label) -> Result<f64, ModelError>;
}