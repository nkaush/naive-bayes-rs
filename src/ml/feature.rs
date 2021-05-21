extern crate num_traits;

use self::num_traits::ToPrimitive;
use crate::ml::label::Label;

pub trait Feature {
    fn get_feature_likelihood_given_class<Num: ToPrimitive + Copy>
        (&self, feature: Num, class: &dyn Label) -> f64;

    fn get_class_likelihood(&self, class: &dyn Label) -> f64;
}