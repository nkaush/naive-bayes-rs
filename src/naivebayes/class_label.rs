extern crate serde;

use self::serde::{Serialize, Deserialize};
use crate::ml::label::Label;

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

    pub(crate) fn get_ascii(&self) -> u8 {
        self.ascii
    }
}

impl Label for ClassLabel {
    fn get_index(&self) -> usize {
        self.index
    }
}