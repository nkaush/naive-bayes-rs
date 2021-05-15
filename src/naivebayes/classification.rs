extern crate num_traits;

use self::num_traits::ToPrimitive;

pub trait<T> Classification<T> {
    fn feature_likelihood_given_class<Num: ToPrimitive + Copy>(&self, feature: Num, &class: T) -> f64;
}