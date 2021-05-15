extern crate num_traits;

use crate::naivebayes::gaussian_feature::GaussianFeature;
use crate::naivebayes::classification::Classification;
use self::num_traits::ToPrimitive;
use std::vec::Vec;

pub struct GaussianClassification<T> {
    labels: Vec<T>,
    features: Vec<GaussianFeature>
}

impl<T> GaussianClassification<T> {
    pub(crate) fn get_class_index<T>(&class: T) -> usize {
        0
    }
}

impl Classification<T> for GaussianClassification<T> {
    pub(crate) fn feature_likelihood_given_class<T>(sample_feature: Num, &class: T) -> f64 {
        let class_idx: usize = self.get_class_index(class);
        let model_feature: GaussianFeature = self.features[class_idx];

        model_feature.pdf(sample_feature)
    }
}