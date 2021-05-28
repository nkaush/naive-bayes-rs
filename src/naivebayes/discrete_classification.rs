extern crate serde;

use crate::ml::error::ModelError;
use self::serde::{Serialize, Deserialize};
use std::{string::String, vec::Vec};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DiscreteClassification {
    sample_size: usize,
    occurrences: Vec<usize>,
    classes: Vec<String>
}

impl DiscreteClassification {
    pub(crate) fn add_occurrence(&mut self, sample: &String) {
        let iter = self.classes.iter().zip(self.occurrences.iter_mut());
        self.sample_size += 1;

        for (class, occurrence) in iter {
            if class == sample {
                *occurrence += 1;
                break;
            }
        }

        self.classes.push(sample.to_string());
        self.occurrences.push(1);
    }

    pub(crate) fn get_num_classes(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn get_class_occurrences(&self, sample: &String) 
            -> Result<usize, ModelError> {
        for (idx, class) in self.classes.iter().enumerate() {
            if class == sample {
                return Ok(self.occurrences[idx]);
            }
        }

        Err(ModelError::FeatureNotFound)
    }
}
