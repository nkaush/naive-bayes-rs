extern crate num_traits;
extern crate serde;

use crate::naivebayes::discrete_classification::DiscreteClassification;
use crate::ml::{feature::Feature, label::Label, error::ModelError};

use std::{string::String, vec::Vec, str::FromStr};
use self::serde::{Serialize, Deserialize};
use num_traits::ToPrimitive;

#[derive(Serialize, Deserialize, Debug)]
pub struct DiscreteFeature {
    is_trained: bool,
    sample_size: usize,
    classifications: Vec<DiscreteClassification>
}

impl DiscreteFeature {
    fn get_class(&self, label: &dyn Label) -> &DiscreteClassification {
        &self.classifications[label.get_index()]
    }

    fn get_class_mut(&mut self, label: &dyn Label) -> &mut DiscreteClassification {
        &mut self.classifications[label.get_index()]
    }
}

impl Feature for DiscreteFeature {
    fn train_iter<Num: ToPrimitive + Copy + FromStr>
            (&mut self, label: &dyn Label, value: &String, iter: usize) {
        match iter {
            0 => {
                // self.get_class_mut(label).add_occurrence(value);
                // let class: &mut GaussianClassification = 
                //     &mut self.classifications[label.get_index()];
                
                // class.add_value_for_mean(value);
                // self.sample_size += 1;
            },
            1 => {
                // let class: &mut GaussianClassification = 
                //     &mut self.classifications[label.get_index()];
                // class.add_value_for_std(value);

            },
            _ => {}
        }
    }

    fn prepare(&mut self) {
        self.is_trained = true
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    fn likelihood_given_class<Num: ToPrimitive + Copy + FromStr>
        (&self, feature: &String, class: &dyn Label) 
            -> Result<f64, ModelError> {
        Ok(0.0)
    }

    fn class_likelihood(&self, class: &dyn Label) -> Result<f64, ModelError> {
        Ok(0.0)
    }
}