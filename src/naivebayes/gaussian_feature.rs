extern crate num_traits;

use crate::naivebayes::gaussian_classification::GaussianClassification;
use crate::naivebayes::feature::{Feature, ClassLabel};
use self::num_traits::ToPrimitive;
use std::vec::Vec;

pub struct GaussianFeature {
    sample_size: usize,
    classifications: Vec<GaussianClassification>
}

impl GaussianFeature {
    pub(crate) fn new(count: usize) -> GaussianFeature {
        GaussianFeature {
            sample_size: 0,
            classifications: (0..count)
                .map(|_| GaussianClassification::new())
                .collect::<Vec<GaussianClassification>>()
        } 
    }

    fn get_class(&self, label: &ClassLabel) -> &GaussianClassification {
        &self.classifications[label.get_index()]
    }

    pub(crate) fn add_value_for_mean<Num: ToPrimitive + Copy>
            (&mut self, label: &ClassLabel, value: Num) {
        let class: &mut GaussianClassification = 
            &mut self.classifications[label.get_index()];
        class.add_value_for_mean(value);
        self.sample_size += 1;
    }

    pub(crate) fn add_value_for_std<Num: ToPrimitive + Copy>
            (&mut self, label: &ClassLabel, value: Num) {
        let class: &mut GaussianClassification = 
            &mut self.classifications[label.get_index()];
        class.add_value_for_std(value);
    }

    pub(crate) fn configure_std(&mut self) {
        for class in self.classifications.iter_mut() {
            class.configure_std();
        }
    }
}

impl Feature for GaussianFeature {
    fn get_feature_likelihood_given_class<Num: ToPrimitive + Copy>
            (&self, sample_feature: Num, label: &ClassLabel) -> f64 {
        self.get_class(label).pdf(sample_feature)
    }

    fn get_class_likelihood(&self, label: &ClassLabel) -> f64 {
        self.get_class(label).get_sample_size() as f64 / self.sample_size as f64
    }
}
