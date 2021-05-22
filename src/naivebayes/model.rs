extern crate num_traits;
extern crate ndarray;
extern crate serde;
extern crate csv;

use std::{error::Error, vec::Vec, string::String};
use std::{fs, io::{BufReader, BufRead}};
use std::{sync::{Arc, Mutex, mpsc}, thread};
use std::sync::mpsc::{Sender, Receiver};

use crate::ml::{model::Model, feature::Feature, label::Label};
use crate::naivebayes::gaussian_feature::GaussianFeature;
use crate::naivebayes::class_label::ClassLabel;

use self::serde::{Serialize, Deserialize};
use self::ndarray::{prelude::*, Array};
use self::num_traits::ToPrimitive;
use core::str::FromStr;

type Matrix<T> = Array<T, Ix2>;

static PRINT_INTERVAL: usize = 5000;

#[derive(Serialize, Deserialize, Debug)]
pub struct GaussianNaiveBayes {
    labels: Vec<ClassLabel>,
    features: Vec<GaussianFeature>
}

impl GaussianNaiveBayes {
    fn add_values_from_file<Num: ToPrimitive + Copy + FromStr>
            (&mut self, file_path: &String, 
                add_fn: &mut dyn FnMut(&mut GaussianFeature, &ClassLabel, Num))
            -> Result<(), Box<dyn Error>> {
        // Build the CSV reader and iterate over each record.
        let mut rdr = csv::Reader::from_path(file_path)?;

        // Get the 1st record wrapped in an Option<Result<>>, panic otherwise
        let record = rdr.records().nth(0).expect("Could not parse CSV record.")?;

        // subtract 1 to skip over the row index label 
        let num_features = record.len() - 1; 
        self.features = (0..num_features)
            .map(|_| GaussianFeature::new(self.labels.len()))
            .collect::<Vec<GaussianFeature>>();

        // Skip the 1st entry since it was handles above
        for (idx, result) in rdr.records().skip(1).enumerate() {
            // iterator yields Result<StringRecord, Error>, so check error here.
            let record = result?;

            // Print the training status on the specified interval
            if idx % PRINT_INTERVAL == 0 {
                println!("Iteration {}", idx);
            }

            // Get the row label from the csv file
            let label_index: usize = record.get(0)
                .expect("Could not parse record.").parse::<usize>()
                .expect("Could not convert String to Num.");
            let label: &ClassLabel = &self.labels[label_index];

            // Skip the 1st element since we already parsed it above
            for (idx, data) in record.iter().skip(1).enumerate() {
                let value: Num = match data.parse::<Num>() {
                    Ok(val) => val,
                    Err(_) => panic!("Could not parse record.")
                };

                // adapted from https://stackoverflow.com/a/37391993
                add_fn(&mut self.features[idx], label, value);
            }
        }

        Ok(())
    }

    fn experiment<Num: ToPrimitive + Copy + FromStr>
            (&self, file_path: &String) -> Result<f64, Box<dyn Error>> {
        
        println!("Testing model.");

        let num_threads: usize = 20;
        let matrix_shape: usize = self.labels.len();

        let model_data: Arc<Vec<GaussianFeature>> = Arc::new(self.features.to_owned());
        let model_labels: Arc<Vec<ClassLabel>> = Arc::new(self.labels.to_owned());
        
        // TODO move path to arc to avoid repeated cloning
        // let csv_path: Arc<String> = 

        let receivers: Vec<Receiver<Matrix<usize>>> = 
            (0..num_threads).map(|thread_idx| {
                let (tx, rx) = mpsc::channel();

                let model_clone: Arc<Vec<GaussianFeature>> = Arc::clone(&model_data);
                let labels_clone: Arc<Vec<ClassLabel>> = Arc::clone(&model_labels);
                let path: String = file_path.clone();
                
                thread::spawn(move || {
                    if let Err(_) = 
                        GaussianNaiveBayes::test_helper::<Num>(
                            model_clone, labels_clone, path, tx, 
                            matrix_shape, thread_idx, num_threads) {
                        println!("Error in Thread {}", thread_idx);
                    }
                });

                rx
            }).collect::<Vec<Receiver<Matrix<usize>>>>();

        for thread_idx in 0..num_threads {
            
        }

        Ok(0.0)
    }

    fn test_helper<Num: ToPrimitive + Copy + FromStr>
            (model: Arc<Vec<GaussianFeature>>, labels: Arc<Vec<ClassLabel>>, 
                csv_file_path: String, transmitter: Sender<Matrix<usize>>, 
                matrix_shape: usize, thread_index: usize, num_threads: usize) 
            -> Result<(), Box<dyn Error>> {
        // let (tx, rx): (mpsc::Sender<_>, mpsc::Receiver<_>) = mpsc::channel();

        let mut rdr = csv::Reader::from_path(csv_file_path)?;
        let csv_iter = rdr.records();

        let mut confusion_matrix: Matrix<usize> = 
            Array::<usize, _>::zeros((matrix_shape, matrix_shape).f());

        for (idx, result) in csv_iter.skip(thread_index).step_by(num_threads).enumerate() {
            // TODO handle printing feedback
            if idx % PRINT_INTERVAL == 0 {
                println!("Thread {}\tIteration {}", thread_index, idx);
            }

            let record = result?;

            let label_index: usize = record.get(0)
                .expect("Could not parse record.").parse::<usize>()
                .expect("Could not convert String to usize.");
            let actual_label: &ClassLabel = &labels[label_index];

            // Skip the 1st element since we already parsed it above
            let features: Vec<Num> = record.iter().skip(1).map(|x| {
                match x.parse::<Num>() {
                    Ok(val) => val,
                    Err(_) => panic!("Could not parse record.")
                }
            }).collect::<Vec<Num>>();

            let predicted_label: Box<dyn Label> = 
                GaussianNaiveBayes::helper_classify::<Num>(&model, &labels, &features);

            let predicted_index: usize = predicted_label.get_index();

            confusion_matrix[[actual_label.get_index(), predicted_index]] += 1;
        }

        transmitter.send(confusion_matrix).unwrap();

        Ok(())
    }

    fn helper_classify<Num: ToPrimitive + Copy>
            (model: &Arc<Vec<GaussianFeature>>, 
                labels: &Arc<Vec<ClassLabel>>,
                sample_features: &Vec<Num>) 
            -> Box<dyn Label> {
        let mut max_likelihood: f64 = 0.0;
        let mut best_label: Option<&ClassLabel> = None;

        for current_class in labels.iter() {        
            let feature_likelihoods: f64 = model.iter().enumerate()
                                            .fold(0.0, |total, (idx, feat)| {
                let prob: f64 = feat.get_feature_likelihood_given_class(
                    sample_features[idx], current_class);
                total + prob.log10()
            });

            let class_likelihood: f64 = 
                model[0].get_class_likelihood(current_class).log10();
            let likelihood: f64 = feature_likelihoods + class_likelihood;

            if best_label.is_none() || likelihood > max_likelihood {
                max_likelihood = likelihood;
                best_label = Some(current_class);
            }
        }

        Box::new(*best_label.expect("Could not classify record."))
    }
}

impl Model for GaussianNaiveBayes {
    fn from_json(file_path: &String) -> GaussianNaiveBayes {
        print!("Loading model from {}...", file_path);
        
        let data = fs::read_to_string(file_path).expect("Unable to read JSON file.");
        let deserialized: GaussianNaiveBayes = 
            serde_json::from_str(&data).expect("Could not deserialize JSON.");
        
        println!("done.");

        deserialized
    }

    fn to_json(&self, file_path: &String) {
        print!("Saving model to {}...", file_path);

        let data = serde_json::to_string(&self).expect("Could not serialize JSON.");
        let message: String = format!("Unable to write to file at {}", file_path);
        fs::write(file_path, data).expect(&message);
        
        println!("done.")
    }

    fn from_labels(file_path: &String) -> GaussianNaiveBayes {
        let file = match fs::File::open(file_path) {
            Ok(f) => f,
            Err(_) => panic!("Unable to read labels file at {}", file_path)
        };

        print!("Reading labels...");

        let reader: BufReader<fs::File> = BufReader::new(file);
        let labels: Vec<ClassLabel> = reader.lines().map(|line| {
            let components: Vec<String> = match line {
                Ok(s) => s.split(" ").map(|s| s.to_string()).collect(),
                Err(_) => panic!("Could not parse label.")
            };

            ClassLabel::new(
                components[0].parse::<usize>().unwrap(),
                components[1].parse::<u8>().unwrap()
            )
        }).collect::<Vec<ClassLabel>>();

        println!("done.");

        GaussianNaiveBayes {
            labels: labels,
            features: Vec::new()
        }
    }

    fn train<Num: ToPrimitive + Copy + FromStr>(
            &mut self, file_path: &String) -> Result<(), Box<dyn Error>> {
        println!("Adding distribution means.");
        self.add_values_from_file::<Num>(
            &file_path, &mut GaussianFeature::add_value_for_mean)?;

        println!("Adding distribution standard deviations.");
        self.add_values_from_file::<Num>(
            &file_path, &mut GaussianFeature::add_value_for_std)?;

        for feature in self.features.iter_mut() {
            feature.configure_std();
        }

        Ok(())
    }

    fn test<Num: ToPrimitive + Copy + FromStr>
            (&self, file_path: &String) -> Result<f64, Box<dyn Error>> {
        println!("Testing model.");
        let mut rdr = csv::Reader::from_path(file_path)?;

        let shape: usize = self.labels.len();
        // Row: actual, Column: predicted
        let mut confusion_matrix: Matrix<usize> = 
            Array::<usize, _>::zeros((shape, shape).f());

        for (idx, result) in rdr.records().enumerate() {
            if idx % PRINT_INTERVAL == 0 {
                println!("Iteration {}", idx);
            }

            // iterator yields Result<StringRecord, Error>, so check error here.
            let record = result?;

            // Get the row label from the csv file
            let label_index: usize = record.get(0)
                .expect("Could not parse record.").parse::<usize>()
                .expect("Could not convert String to usize.");
            let actual_label: &ClassLabel = &self.labels[label_index];

            // Skip the 1st element since we already parsed it above
            let features: Vec<Num> = record.iter().skip(1).map(|x| {
                match x.parse::<Num>() {
                    Ok(val) => val,
                    Err(_) => panic!("Could not parse record.")
                }
            }).collect::<Vec<Num>>();

            let predicted_label: Box<dyn Label> = self.classify::<Num>(&features);
            let predicted_index: usize = predicted_label.get_index();

            confusion_matrix[[actual_label.get_index(), predicted_index]] += 1;
        }

        println!("Confusion Matrix:");
        println!("{:?}\n", confusion_matrix);

        Ok(confusion_matrix.diag().sum() as f64 / confusion_matrix.sum() as f64)
    }

    fn classify<Num: ToPrimitive + Copy>
            (&self, sample_features: &Vec<Num>) -> Box<dyn Label> {
        let mut max_likelihood: f64 = 0.0;
        let mut best_label: Option<&ClassLabel> = None;

        for current_class in self.labels.iter() {        
            let feature_likelihoods: f64 = self.features
                                            .iter().enumerate()
                                            .fold(0.0, |total, (idx, feat)| {
                let prob: f64 = feat.get_feature_likelihood_given_class(
                    sample_features[idx], current_class);
                total + prob.log10()
            });

            let class_likelihood: f64 = 
                self.features[0].get_class_likelihood(current_class).log10();
            let likelihood: f64 = feature_likelihoods + class_likelihood;

            if best_label.is_none() || likelihood > max_likelihood {
                max_likelihood = likelihood;
                best_label = Some(current_class);
            }
        }

        Box::new(*best_label.expect("Could not classify record."))
    }
}
