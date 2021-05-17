extern crate num_traits;

use crate::naivebayes::gaussian_classification::GaussianClassification;
use crate::naivebayes::feature::{Feature, ClassLabel};
use self::num_traits::ToPrimitive;
use std::vec::Vec;

pub struct GaussianFeature {
    labels: Vec<ClassLabel>,
    sample_size: usize,
    classifications: Vec<GaussianClassification>
}

impl GaussianFeature {
}

impl Feature for GaussianFeature {
    fn get_feature_likelihood_given_class<Num: ToPrimitive + Copy>
            (&self, sample_feature: Num, label: &ClassLabel) -> f64 {
        let class: &GaussianClassification = 
            &self.classifications[label.get_index()];

        class.pdf(sample_feature)
    }

    fn get_class_likelihood(&self, label: &ClassLabel) -> f64 {
        let class: &GaussianClassification = 
            &self.classifications[label.get_index()];

        class.get_sample_size() as f64 / self.sample_size as f64
    }
}
