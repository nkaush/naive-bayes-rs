extern crate num_traits;
extern crate serde;

use crate::naivebayes::gaussian_classification::GaussianClassification;
use crate::ml::{feature::Feature, label::Label};
use crate::naivebayes::class_label::ClassLabel;

use self::serde::{Serialize, Deserialize};
use self::num_traits::ToPrimitive;
use std::vec::Vec;

#[derive(Serialize, Deserialize, Debug)]
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

    fn get_class(&self, label: &dyn Label) -> &GaussianClassification {
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
            (&self, sample_feature: Num, label: &dyn Label) -> f64 {
        self.get_class(label).pdf(sample_feature)
    }

    fn get_class_likelihood(&self, label: &dyn Label) -> f64 {
        self.get_class(label).get_sample_size() as f64 / self.sample_size as f64
    }
}
