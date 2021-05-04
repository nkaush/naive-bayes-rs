use crate::naivebayes::training_image::TrainingImage;
use std::collections::HashMap;
use std::vec::Vec;

pub struct Dataset {
    size: usize,
    groups: HashMap<char, Vec<TrainingImage>>
}

impl Dataset {
    pub(crate) fn new() -> Dataset {
        Dataset {
            size: 0,
            groups: Default::default()
        }
    }
}
