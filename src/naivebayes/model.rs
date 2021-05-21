extern crate num_traits;
extern crate ndarray;
extern crate csv;

use std::{error::Error, vec::Vec, string::String, fs::File};
use crate::naivebayes::gaussian_feature::GaussianFeature;
use crate::naivebayes::feature::{Feature, ClassLabel};
use std::io::{BufReader, BufRead};
use self::num_traits::ToPrimitive;
use self::ndarray::prelude::*;
use self::ndarray::Array;
use core::str::FromStr;

static PRINT_INTERVAL: usize = 5000;

pub struct GaussianNaiveBayes {
    labels: Vec<ClassLabel>,
    features: Vec<GaussianFeature>
}

impl GaussianNaiveBayes {
    pub(crate) fn from_labels(file_path: &String) -> GaussianNaiveBayes {
        let file = match File::open(file_path) {
            Ok(f) => f,
            Err(_) => panic!("NOT FOUND")
        };

        print!("Reading labels...");

        let mut labels: Vec<ClassLabel> = Vec::new();
        let reader = BufReader::new(file);

        for line_buffer in reader.lines() {
            let components: Vec<String> = match line_buffer {
                Ok(s) => s.split(" ").map(|s| s.to_string()).collect(),
                Err(_) => panic!("Could not parse label.")
            };

            let label: ClassLabel = ClassLabel::new(
                components[0].parse::<usize>().unwrap(),
                components[1].parse::<u8>().unwrap()
            );

            labels.push(label);
        }

        println!("done.");

        GaussianNaiveBayes {
            labels: labels,
            features: Vec::new()
        }
    }

    fn add_values_from_file<Num: ToPrimitive + Copy + FromStr>
            (&mut self, file_path: &String, 
             add_fn: &mut dyn FnMut(&mut GaussianFeature, &ClassLabel, Num))
            -> Result<(), Box<dyn Error>> {
        // Build the CSV reader and iterate over each record.
        let mut rdr = csv::Reader::from_path(file_path)?;

        for (idx, result) in rdr.records().enumerate() {
            // The iterator yields Result<StringRecord, Error>, so we check the error here.
            let record = result?;

            if idx % PRINT_INTERVAL == 0 {
                println!("Iteration {}", idx);
            }

            if self.features.is_empty() {
                let num_features = record.len() - 1; // subtract 1 for the row index
                self.features = (0..num_features)
                    .map(|_| GaussianFeature::new(self.labels.len()))
                    .collect::<Vec<GaussianFeature>>();
            }

            // Get the row label from the csv file
            let label_index: usize = record.get(0).unwrap().parse::<usize>().unwrap();
            let label: &ClassLabel = &self.labels[label_index];

            // Skip the 1st element since we already parsed it above
            for (idx, data) in record.iter().skip(1).enumerate() {
                let value: Num = match data.parse::<Num>() {
                    Ok(val) => val,
                    Err(_) => panic!("Could not parse record.")
                };

                // adapted from https://stackoverflow.com/a/37391993
                let feature: &mut GaussianFeature = &mut self.features[idx];
                add_fn(feature, label, value);
            }
        }
        Ok(())
    }

    pub(crate) fn train<Num: ToPrimitive + Copy + FromStr>(
            &mut self, file_path: &String) -> Result<(), Box<dyn Error>> {
        println!("Adding distribution means.");
        self.add_values_from_file::<Num>(&file_path, &mut GaussianFeature::add_value_for_mean)?;

        println!("Adding distribution standard deviations.");
        self.add_values_from_file::<Num>(&file_path, &mut GaussianFeature::add_value_for_std)?;

        for feature in self.features.iter_mut() {
            feature.configure_std();
        }

        Ok(())
    }

    pub(crate) fn test<Num: ToPrimitive + Copy + FromStr>
            (&self, file_path: &String) -> Result<f64, Box<dyn Error>> {
        println!("Testing model.");
        let mut rdr = csv::Reader::from_path(file_path)?;

        let shape: usize = self.labels.len();
        // Row: actual, Column: predicted
        let mut confusion_matrix = Array::<usize, _>::zeros((shape, shape).f());

        for (idx, result) in rdr.records().enumerate() {
            if idx % PRINT_INTERVAL == 0 {
                println!("Iteration {}", idx);
            }

            // The iterator yields Result<StringRecord, Error>, so we check the error here.
            let record = result?;

            // Get the row label from the csv file
            let label_index: usize = record.get(0).unwrap().parse::<usize>().unwrap();
            let actual_label: &ClassLabel = &self.labels[label_index];

            // Skip the 1st element since we already parsed it above
            let features: Vec<Num> = record.iter().skip(1).map(|x| {
                match x.parse::<Num>() {
                    Ok(val) => val,
                    Err(_) => panic!("Could not parse record.")
                }
            }).collect::<Vec<Num>>();

            let predicted_label: ClassLabel = match self.classify::<Num>(&features) {
                Ok(label) => label,
                Err(_) => panic!("Could not classify record.")
            };

            confusion_matrix[[actual_label.get_index(), predicted_label.get_index()]] += 1;
        }

        println!("Confusion Matrix:");
        println!("{:?}\n", confusion_matrix);

        Ok(confusion_matrix.diag().sum() as f64 / confusion_matrix.sum() as f64)
    }

    pub(crate) fn classify<Num: ToPrimitive + Copy>
            (&self, sample_features: &Vec<Num>) -> Result<ClassLabel, ()> {
        let mut max_likelihood: f64 = 0.0;
        let mut best_label: Option<&ClassLabel> = None;

        for current_class in self.labels.iter() {        
            let feature_likelihoods: f64 = self.features
                                            .iter().enumerate()
                                            .fold(0.0, |total, (idx, feat)| {
                let prob: f64 = feat.get_feature_likelihood_given_class(
                    sample_features[idx], &current_class);
                total + prob.log10()
            });

            let class_likelihood: f64 = 
                self.features[0].get_class_likelihood(&current_class).log10();
            let likelihood: f64 = feature_likelihoods + class_likelihood;

            best_label = match best_label {
                None => {
                    max_likelihood = likelihood;
                    Some(&current_class)
                },
                Some(previous_label) => {
                    if likelihood > max_likelihood {
                        max_likelihood = likelihood;
                        Some(&current_class)
                    } else {
                        Some(previous_label)
                    }
                }
            };
        }
        
        match best_label {
            None => Err(()),
            Some(label) => Ok(*label)
        }
    }
}
