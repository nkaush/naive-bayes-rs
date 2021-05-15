extern crate num_traits;

use self::num_traits::ToPrimitive;
use std::f64::consts::PI;
use std::vec::Vec;

#[derive(Debug, Default)]
pub struct GaussianClassification {
    mean: f64,
    std: f64
}

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
    pub(crate) fn new<Num: ToPrimitive + Copy>(features: Vec<Num>) -> GaussianClassification {
        GaussianClassification {
            mean: mean(&features),
            std: std(&features)
        }
    }

    pub(crate) fn create<Num: ToPrimitive + Copy>(mean: f64, std: f64) -> GaussianClassification {
        GaussianClassification {
            mean,
            std
        }
    }

    pub(crate) fn pdf<Num: ToPrimitive>(&self, x: Num) -> f64 {
        match x.to_f64() {
            None => 0.0,
            Some(n) => {
                let multiplier: f64 = 1.0 / (self.std.powf(2.0) * 2.0 * PI);
                let exponent: f64 = -0.5 * ((n - self.mean) / self.std).powf(2.0);

                multiplier * exponent.exp()
            }
        }
    }
}
