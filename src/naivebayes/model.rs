use crate::naivebayes::gaussian_feature::GaussianFeature;
use std::vec::Vec;

pub struct Model {

}

pub struct GaussianNaiveBayes {
    features: Vec<GaussianFeature>
}

impl Model {
    pub(crate) fn new(smoothing: u8) -> Model {
        return Model {

        };
    }

    pub(crate) fn train(&self) {

    }

    pub(crate) fn test(&self) {

    }

    pub(crate) fn classify(&self) -> char {
        '0'
    }
}
