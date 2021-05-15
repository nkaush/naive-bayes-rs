extern crate num_traits;

use self::num_traits::ToPrimitive;

pub struct ClassLabel {
    index: usize,
    ascii: u8
}

impl ClassLabel {
    pub(crate) fn get_index(&self) -> usize {
        self.index
    }

    pub(crate) fn get_ascii(&self) -> u8 {
        self.ascii
    }
}

pub trait Classification<T> {
    fn feature_likelihood_given_class<Num: ToPrimitive + Copy>(&self, feature: Num, class: &ClassLabel) -> f64;
}