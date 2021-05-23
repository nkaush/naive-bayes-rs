use std::{error::Error, fmt};

#[derive(Debug)]
pub enum ModelError {
    UntrainedError
}

impl ModelError {
    fn value(&self) -> &str {
        match *self {
            ModelError::UntrainedError => "The model has not been fully trained."
        }
    }
}

impl Error for ModelError {
    fn description(&self) -> &str {
        self.value()
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}
