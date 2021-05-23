extern crate num_traits;

use crate::ml::{label::Label, error::ModelError};
use self::num_traits::ToPrimitive;

pub trait Feature {
    fn train_iter<Num: ToPrimitive + Copy>
        (&mut self, label: &dyn Label, value: Num, iter: usize);

    fn prepare(&mut self);

    fn is_trained(&self) -> bool;

    fn get_feature_likelihood_given_class<Num: ToPrimitive + Copy>
        (&self, feature: Num, class: &dyn Label) -> Result<f64, ModelError>;

    fn get_class_likelihood(&self, class: &dyn Label) -> Result<f64, ModelError>;
}