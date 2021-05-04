pub struct Matrix<T> {
    values: Vec<Vec<T>>
}

impl<T> Matrix<T> {
    pub(crate) fn get_width(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn get_height(&self) -> usize {
        match self.get_width() {
            0 => 0,
            _ => self.values[0].len()
        }
    }
}