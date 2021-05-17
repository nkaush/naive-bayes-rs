extern crate num_traits;

use crate::naivebayes::gaussian_feature::GaussianFeature;
use crate::naivebayes::feature::{Feature, ClassLabel};
use self::num_traits::ToPrimitive;
use std::vec::Vec;

pub struct GaussianNaiveBayes {
    label_mappings: Vec<ClassLabel>,
    features: Vec<GaussianFeature>
}

impl GaussianNaiveBayes {
    pub(crate) fn new() -> GaussianNaiveBayes {
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
            let current_class: &ClassLabel = &self.label_mappings[class_index];
            
            let feature_likelihoods: f64 = self.features
                                            .iter().enumerate()
                                            .fold(0.0, |total, (idx, feat)| {
                let prob: f64 = feat.get_feature_likelihood_given_class(
                    sample_features[idx], current_class);
                total + prob.log10()
            });

            let class_likelihood: f64 = 
                self.features[0].get_class_likelihood(current_class).log10();
            let likelihood: f64 = feature_likelihoods + class_likelihood;

            best_label = match best_label {
                None => Some(current_class),
                Some(previous_label) => {
                    if likelihood > max_likelihood {
                        max_likelihood = likelihood;
                        Some(current_class)
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
