extern crate num_traits;

use crate::naivebayes::gaussian_classification::GaussianClassification;
use crate::naivebayes::classification::{Classification, ClassLabel};
use self::num_traits::ToPrimitive;
use std::vec::Vec;

pub struct GaussianFeature<T> {
    labels: Vec<T>,
    classifications: Vec<GaussianClassification>
}

impl<T> GaussianFeature<T> {
    
}

impl<T> Classification<T> for GaussianFeature<T> {
    fn feature_likelihood_given_class<Num: ToPrimitive + Copy>
            (&self, sample_feature: Num, label: &ClassLabel) -> f64 {
        let model_feature: &GaussianClassification = &self.classifications[label.get_index()];

        model_feature.pdf(sample_feature)
    }
}
