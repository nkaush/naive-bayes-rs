extern crate num_traits;
extern crate serde;

use crate::naivebayes::gaussian_classification::GaussianClassification;
use crate::ml::{feature::Feature, label::Label, error::ModelError};

use self::serde::{Serialize, Deserialize};
use self::num_traits::ToPrimitive;
use std::vec::Vec;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GaussianFeature {
    is_trained: bool,
    sample_size: usize,
    classifications: Vec<GaussianClassification>
}

impl GaussianFeature {
    pub(crate) fn merge(a: &GaussianFeature, b: &GaussianFeature) 
            -> GaussianFeature {
        let (av, bv) = (&a.classifications, &b.classifications);

        GaussianFeature {
            is_trained: false,
            sample_size: a.sample_size + b.sample_size,
            classifications: (0..a.classifications.len())
                .map(|idx| GaussianClassification::merge(&av[idx], &bv[idx]))
                .collect::<Vec<GaussianClassification>>()
        } 
    }

    pub(crate) fn new(count: usize) -> GaussianFeature {
        GaussianFeature {
            is_trained: false,
            sample_size: 0,
            classifications: (0..count)
                .map(|_| GaussianClassification::new())
                .collect::<Vec<GaussianClassification>>()
        } 
    }

    fn get_class(&self, label: &dyn Label) -> &GaussianClassification {
        &self.classifications[label.get_index()]
    }

    /* pub(crate) fn add_value_for_mean<Num: ToPrimitive + Copy>
            (&mut self, label: &dyn Label, value: Num) {
        let class: &mut GaussianClassification = 
            &mut self.classifications[label.get_index()];
        class.add_value_for_mean(value);
        self.sample_size += 1;
    }

    pub(crate) fn add_value_for_std<Num: ToPrimitive + Copy>
            (&mut self, label: &dyn Label, value: Num) {
        let class: &mut GaussianClassification = 
            &mut self.classifications[label.get_index()];
        class.add_value_for_std(value);
    } 

    pub(crate) fn configure_std(&mut self) {
        for class in self.classifications.iter_mut() {
            class.configure_std();
        }

        self.is_trained = true;
    } */
}

impl Feature for GaussianFeature {
    fn train_iter<Num: ToPrimitive + Copy>
            (&mut self, label: &dyn Label, value: Num, iter: usize) {
        match iter {
            0 => {
                let class: &mut GaussianClassification = 
                    &mut self.classifications[label.get_index()];
                
                class.add_value_for_mean(value);
                self.sample_size += 1;
            },
            1 => {
                let class: &mut GaussianClassification = 
                    &mut self.classifications[label.get_index()];
                class.add_value_for_std(value);
            },
            _ => {}
        }
    }

    fn prepare(&mut self) {
        for class in self.classifications.iter_mut() {
            class.configure_std();
        }

        self.is_trained = true;
    }

    fn is_trained(&self) -> bool {
        self.is_trained
    }

    fn get_feature_likelihood_given_class<Num: ToPrimitive + Copy>
            (&self, sample_feature: Num, label: &dyn Label) 
            -> Result<f64, ModelError> {
        if !self.is_trained() {
            return Err(ModelError::UntrainedError);
        }
        
        Ok(self.get_class(label).pdf(sample_feature))
    }

    fn get_class_likelihood(&self, label: &dyn Label) -> Result<f64, ModelError> {
        if !self.is_trained() {
            return Err(ModelError::UntrainedError);
        }

        let class_size: f64 = self.get_class(label).get_sample_size() as f64;
        Ok(class_size / self.sample_size as f64)
    }
}
