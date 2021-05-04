use crate::naivebayes::matrix::Matrix;
use std::collections::HashMap;

pub struct Model {
    laplace_smoothing: u8,
    labels: Vec<char>,
    label_indices: HashMap<char, usize>,
    class_counts: Vec<u32>,
    class_likelihoods: Vec<f64>,
    feature_counts: Vec<Vec<Matrix<u32>>>,
    feature_likelihoods: Vec<Vec<Matrix<f64>>>
}

impl Model {
    pub(crate) fn new(smoothing: u8) -> Model {
        return Model {
            laplace_smoothing: smoothing,
            labels: vec![],
            label_indices: HashMap::new(),
            class_counts: vec![],
            class_likelihoods: vec![],
            feature_counts: vec![],
            feature_likelihoods: vec![]
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
