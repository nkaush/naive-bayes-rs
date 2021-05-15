extern crate num_traits;

use crate::naivebayes::classification::{Classification, ClassLabel};
use crate::naivebayes::gaussian_feature::GaussianFeature;
use self::num_traits::ToPrimitive;
use std::vec::Vec;

pub struct GaussianNaiveBayes<T> {
    label_mappings: Vec<ClassLabel>,
    features: Vec<GaussianFeature<T>>
}

impl<T> GaussianNaiveBayes<T> {
    pub(crate) fn new() -> GaussianNaiveBayes<T> {
        GaussianNaiveBayes {
            label_mappings: vec![],
            features: vec![]
        }
    }

    pub(crate) fn train(&self) {

    }

    pub(crate) fn test(&self) {

    }

    pub(crate) fn classify<Num: ToPrimitive + Copy>
            (&self, sample_features: &Vec<Num>) -> Result<char, ()> {
        let mut max_likelihood: f64 = 0.0;
        let mut best_label: Option<&ClassLabel> = None;

        for class_index in 0..self.features.len() {
            let current_label: &ClassLabel = &self.label_mappings[class_index];
            
            let feature_likelihoods: f64 = self.features
                                            .iter()
                                            .enumerate()
                                            .fold(0.0, |total, (idx, feature)| {
                let prob: f64 = feature.feature_likelihood_given_class(
                    sample_features[idx], current_label);
                total + prob.log10()
            });

            // TODO: add class likelihood

            let likelihood = feature_likelihoods + 0.0;

            best_label = match best_label {
                None => Some(current_label),
                Some(previous_label) => {
                    if likelihood > max_likelihood {
                        max_likelihood = likelihood;
                        Some(current_label)
                    } else {
                        Some(previous_label)
                    }
                }
            };
        }
        
        match best_label {
            None => Err(()),
            Some(label) => Ok(label.get_ascii() as char)
        }
    }
}
