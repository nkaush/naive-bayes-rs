extern crate num_traits;
extern crate serde;

use self::serde::{Serialize, Deserialize};
use self::num_traits::ToPrimitive;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct ClassLabel {
    index: usize,
    ascii: u8
}

impl ClassLabel {
    pub(crate) fn new(index: usize, ascii: u8) -> ClassLabel {
        ClassLabel {
            index,
            ascii
        }
    }

    pub(crate) fn get_index(&self) -> usize {
        self.index
    }

    pub(crate) fn get_ascii(&self) -> u8 {
        self.ascii
    }
}

pub trait Feature {
    fn get_feature_likelihood_given_class<Num: ToPrimitive + Copy>
        (&self, feature: Num, class: &ClassLabel) -> f64;

    fn get_class_likelihood(&self, class: &ClassLabel) -> f64;
}