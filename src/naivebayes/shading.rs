pub enum Shading {
    Black(u8),
    Gray(u8),
    White(u8)
}

impl Shading {
    pub(crate) fn to_int(&self) -> u8 {
        match &self {
            Shading::Black(_) => 2,
            Shading::Gray(_) => 1,
            Shading::White(_) => 0
        }
    }
}