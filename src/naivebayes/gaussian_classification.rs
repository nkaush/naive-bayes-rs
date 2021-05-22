extern crate num_traits;
extern crate serde;

use self::serde::{Serialize, Deserialize};
use self::num_traits::ToPrimitive;
use std::f64::consts::PI;
use std::vec::Vec;

static MIN_STD: f64 = 1e-10;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GaussianClassification {
    mean: f64,
    std: f64,
    sample_size: usize,
    square_mean_diffs: f64
}

#[allow(dead_code)]
fn mean<Num: ToPrimitive + Copy>(features: &Vec<Num>) -> f64 {
    features.iter().enumerate().fold(0.0, |avg, (i, &x)|
        {
            let value: f64 = match x.to_f64() {
                None => avg,
                Some(n) => n
            };

            avg + ((value - avg) / (i + 1) as f64)
        })
}

#[allow(dead_code)]
fn std<Num: ToPrimitive + Copy>(features: &Vec<Num>) -> f64 {
    let mut sample_size: usize = features.len();
    let mean: f64 = mean(features);

    let sum: f64 = features
        .iter()
        .map(|x: &Num| {
            match x.to_f64() {
                None => {
                    sample_size -= 1;
                    0.0
                }
                Some(n) => (n - mean).powf(2.0)
            }
        })
        .sum::<f64>();

    (sum / sample_size as f64).sqrt()
}

impl GaussianClassification {
    pub(crate) fn new() -> GaussianClassification {
        GaussianClassification {
            mean: 0.0,
            std: 0.0,
            sample_size: 0,
            square_mean_diffs: 0.0
        }
    }

    #[allow(dead_code)]
    pub(crate) fn create<Num: ToPrimitive + Copy>(mean: f64, std: f64) -> GaussianClassification {
        GaussianClassification {
            mean,
            std,
            sample_size: 0,
            square_mean_diffs: 0.0
        }
    }

    pub(crate) fn get_sample_size(&self) -> usize {
        self.sample_size
    }

    pub(crate) fn add_value_for_mean<Num: ToPrimitive + Copy>(&mut self, value: Num) {
        match value.to_f64() {
            None => (),
            Some(n) => {
                self.sample_size += 1;
                self.mean = self.mean + ((n - self.mean) / (self.sample_size) as f64);
            }
        };

        self.sample_size += 1;
    }

    pub(crate) fn add_value_for_std<Num: ToPrimitive + Copy>(&mut self, value: Num) {
        match value.to_f64() {
            None => (),
            Some(n) => {
                self.square_mean_diffs += (n - self.mean).powf(2.0);
            }
        }
    }

    pub(crate) fn configure_std(&mut self) {
        let std_dev: f64 = 
            (self.square_mean_diffs / (self.sample_size - 1) as f64).sqrt();
        
        self.std = if std_dev < MIN_STD { MIN_STD } else { std_dev };
    }

    pub(crate) fn pdf<Num: ToPrimitive>(&self, x: Num) -> f64 {
        match x.to_f64() {
            None => 0.0,
            Some(n) => {
                let multiplier: f64 = 1.0 / (self.std * (2.0 * PI).sqrt());
                let exponent: f64 = -0.5 * ((n - self.mean) / self.std).powf(2.0);

                multiplier * exponent.exp()
            }
        }
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod gaussian_classification_tests {
    use crate::naivebayes::gaussian_classification::GaussianClassification;

    #[test]
    fn test_standard_normal_distribution() {
        let gc: GaussianClassification = GaussianClassification::create::<f64>(0.0, 1.0);
        
        assert_relative_eq!(gc.pdf(0.0), 0.39894228, max_relative=1.0);

        assert_relative_eq!(gc.pdf(1.0), 0.24197072, max_relative=1.0);
        assert_relative_eq!(gc.pdf(-1.0), 0.24197072, max_relative=1.0);

        assert_relative_eq!(gc.pdf(1.5), 0.12951760, max_relative=1.0);
        assert_relative_eq!(gc.pdf(-1.5), 0.12951760, max_relative=1.0);

        assert_relative_eq!(gc.pdf(2.0), 0.05399097, max_relative=1.0);
        assert_relative_eq!(gc.pdf(-2.0), 0.05399097, max_relative=1.0);

        assert_relative_eq!(gc.pdf(2.5), 0.0175283, max_relative=1.0);
        assert_relative_eq!(gc.pdf(-2.5), 0.0175283, max_relative=1.0);
    }

    #[test]
    fn test_arbitrary_normal_distribution() {
        let gc: GaussianClassification = GaussianClassification::create::<f64>(0.0, 1.0);

        assert_relative_eq!(gc.pdf(2.5), 0.0175283, max_relative=1.0);
        assert_relative_eq!(gc.pdf(-2.5), 0.0175283, max_relative=1.0);
    }
}
