use crate::naivebayes::shading::Shading;
use crate::naivebayes::matrix::Matrix;

pub struct TrainingImage {
    label: char,
    pixels: Matrix<Shading>
}

impl TrainingImage {
    pub(crate) fn get_width(&self) -> usize {
        self.pixels.get_width()
    }

    pub(crate) fn get_height(&self) -> usize {
        self.pixels.get_height()
    }
}

