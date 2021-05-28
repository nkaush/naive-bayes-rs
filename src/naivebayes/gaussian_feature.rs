extern crate num_traits;
extern crate serde;

use crate::naivebayes::gaussian_classification::GaussianClassification;
use crate::ml::{feature::Feature, label::Label, error::ModelError};

use self::serde::{Serialize, Deserialize};
use self::num_traits::ToPrimitive;
use std::{str::FromStr, vec::Vec};

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

    fn get_class_mut(&mut self, label: &dyn Label) -> &mut GaussianClassification {
        &mut self.classifications[label.get_index()]
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
    fn train_iter<Num: ToPrimitive + Copy + FromStr>
            (&mut self, label: &dyn Label, value: &String, iter: usize) {
        let converted: Num = match value.parse::<Num>() {
            Ok(val) => val,
            Err(_) => panic!("Could not parse record.")
        };
        
        match iter {
            0 => {
                self.get_class_mut(label).add_value_for_mean(converted);
                self.sample_size += 1;
            },
            1 => {
                self.get_class_mut(label).add_value_for_std(converted);
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

    fn likelihood_given_class<Num: ToPrimitive + Copy + FromStr>
            (&self, sample_feature: &String, label: &dyn Label) 
            -> Result<f64, ModelError> {
        if !self.is_trained() {
            return Err(ModelError::UntrainedError);
        }

        let converted: Num = match sample_feature.parse::<Num>() {
            Ok(val) => val,
            Err(_) => panic!("Could not parse record.")
        };
        
        Ok(self.get_class(label).pdf(converted))
    }

    fn class_likelihood(&self, label: &dyn Label) -> Result<f64, ModelError> {
        if !self.is_trained() {
            return Err(ModelError::UntrainedError);
        }

        let class_size: f64 = self.get_class(label).get_sample_size() as f64;
        Ok(class_size / self.sample_size as f64)
    }
}
