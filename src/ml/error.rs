use std::{error::Error, fmt};

#[derive(Debug)]
pub enum ModelError {
    UntrainedError,
    FeatureNotFound
}

impl ModelError {
    fn value(&self) -> &str {
        match *self {
            ModelError::UntrainedError => "The model has not been fully trained.",
            ModelError::FeatureNotFound => "The feature was not present in the provided training data."
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
