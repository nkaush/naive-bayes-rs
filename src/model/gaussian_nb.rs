extern crate num_traits;
extern crate ndarray;
extern crate serde;
extern crate csv;

use std::sync::{Arc, mpsc, mpsc::{Sender, Receiver}};
use std::{error::Error, vec::Vec, string::String};
use std::{io::{BufReader, BufRead}, fs, thread};

use crate::ml::{model::Model, feature::Feature, label::Label, error::ModelError};
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
    fn parse_csv_record<Num: ToPrimitive + Copy + FromStr>
            (record: Result<csv::StringRecord, csv::Error>) 
            -> Result<(usize, Vec<Num>), Box<dyn Error>> {
        // iterator yields Result<StringRecord, Error>, so check error here.
        let result = record?;

        // Parse the 1st entry to get the image label
        let label_index: usize = result.get(0)
            .expect("Could not parse record.").parse::<usize>()
            .expect("Could not convert String to usize.");

        // Skip the 1st element since we already parsed it above
        let features: Vec<Num> = result.iter().skip(1).map(|x| {
            match x.parse::<Num>() {
                Ok(val) => val,
                Err(_) => panic!("Could not parse record.")
            }
        }).collect::<Vec<Num>>();

        Ok((label_index, features))
    }

    fn add_values_from_file<Num: ToPrimitive + Copy + FromStr>
            (&mut self, file_path: &String, train_iteration: usize)
            -> Result<(), Box<dyn Error>> {
        // Build the CSV reader and iterate over each record.
        let mut rdr = csv::Reader::from_path(file_path)?;

        // Get the 1st record wrapped in an Option<Result<>>, panic otherwise
        let record = rdr.records().nth(0).expect("Could not parse CSV record.")?;

        // subtract 1 to skip over the row index label 
        if train_iteration == 0 {
            let num_features = record.len() - 1; 
            self.features = (0..num_features)
                .map(|_| GaussianFeature::new(self.labels.len()))
                .collect::<Vec<GaussianFeature>>();
        }
        
        for (sample_idx, result) in rdr.records().enumerate() {
            // Print the training status on the specified interval
            if sample_idx % PRINT_INTERVAL == 0 {
                println!("Iteration {}", sample_idx);
            }

            let (label_index, sample): (usize, Vec<Num>) = 
                GaussianNaiveBayes::parse_csv_record::<Num>(result)?;
            let label: &ClassLabel = &self.labels[label_index];

            for (value, feature) in sample.iter().zip(self.features.iter_mut()) {
                feature.train_iter(label, *value, train_iteration);
            }    
        }

        Ok(())
    }

    fn test_linear<Num: ToPrimitive + Copy + FromStr>
            (&self, file_path: &String) -> Result<f64, Box<dyn Error>> {
        println!("Testing model.");
        let mut rdr = csv::Reader::from_path(file_path)?;

        let shape: usize = self.labels.len();
        // Row: actual, Column: predicted
        let mut confusion_matrix: Matrix<usize> = 
            Array::<usize, Ix2>::zeros((shape, shape).f());

        for (idx, result) in rdr.records().enumerate() {
            if idx % PRINT_INTERVAL == 0 {
                println!("Iteration {}", idx);
            }

            let (actual_index, features): (usize, Vec<Num>) = 
                GaussianNaiveBayes::parse_csv_record::<Num>(result)?;

            // Classify the features in the record
            let predicted_label: Box<dyn Label> = self.classify::<Num>(&features)?;
            let predicted_index: usize = predicted_label.get_index();

            // Indexed by row = actual label, column = predicted label
            confusion_matrix[[actual_index, predicted_index]] += 1;
        }

        println!("Confusion Matrix:");
        println!("{:?}\n", confusion_matrix);

        Ok(GaussianNaiveBayes::calculate_accuracy(&confusion_matrix))
    }

    fn test_multithreaded<Num: ToPrimitive + Copy + FromStr>
            (&self, file_path: &String) -> Result<f64, Box<dyn Error>> {
        
        println!("Testing model.");

        let num_threads: usize = 10;
        let matrix_shape: usize = self.labels.len();

        // Wrap copies of the features and labels of the model in Arc pointers 
        // so we can share the model data between threads
        let model_data: Arc<Vec<GaussianFeature>> = Arc::new(self.features.to_owned());
        let model_labels: Arc<Vec<ClassLabel>> = Arc::new(self.labels.to_owned());
        
        // TODO move path to arc to avoid repeated cloning
        // let csv_path: Arc<String> = 

        let receivers: Vec<Receiver<Matrix<usize>>> = 
            (0..num_threads).map(|thread_idx| {
                let (tx, rx) = mpsc::channel();

                // Clone the arc reference so we can pass the same data 
                let model_clone: Arc<Vec<GaussianFeature>> = Arc::clone(&model_data);
                let labels_clone: Arc<Vec<ClassLabel>> = Arc::clone(&model_labels);
                let path: String = file_path.clone();
                
                // Spawn a thread and send the transmitter to pass the results
                thread::spawn(move || {
                    if let Err(_) = 
                        GaussianNaiveBayes::test_helper::<Num>(
                            model_clone, labels_clone, path, tx, 
                            matrix_shape, thread_idx, num_threads) {
                        println!("Error in Thread {}", thread_idx);
                    }
                });

                rx // keep the receiver to collect the results
            }).collect::<Vec<Receiver<Matrix<usize>>>>();

        let zeros_matrix: Matrix<usize> = 
            Array::<usize, _>::zeros((matrix_shape, matrix_shape).f());
            
        // Combine received confusion matrices from each thread into one matrix
        let confusion_matrix: Array<usize, Ix2> = 
            receivers.iter().fold(zeros_matrix, |acc, rx| {
                let matrix: Array<usize, Ix2> = rx.recv().unwrap();
                acc + matrix
            });

        Ok(GaussianNaiveBayes::calculate_accuracy(&confusion_matrix))
    }

    fn test_helper<Num: ToPrimitive + Copy + FromStr>
            (model: Arc<Vec<GaussianFeature>>, labels: Arc<Vec<ClassLabel>>, 
                csv_file_path: String, transmitter: Sender<Matrix<usize>>, 
                matrix_shape: usize, thread_index: usize, num_threads: usize) 
            -> Result<(), Box<dyn Error>> {
        let mut rdr = csv::Reader::from_path(csv_file_path)?;

        // Iterate through the records, starting on offset given by thread index
        // Step by num threads since parallel threads will handle skipped records
        let csv_iter = rdr.records().skip(thread_index).step_by(num_threads);

        let mut confusion_matrix: Matrix<usize> = 
            Array::<usize, _>::zeros((matrix_shape, matrix_shape).f());

        for (idx, result) in csv_iter.enumerate() {
            if idx % 2500 == 0 {
                println!("Thread {}\tIteration {}", thread_index, idx);
            }

            let (actual_index, features): (usize, Vec<Num>) = 
                GaussianNaiveBayes::parse_csv_record::<Num>(result)?;

            let predicted_label: Box<dyn Label> = 
                GaussianNaiveBayes::classify_helper::<Num>(&model, &labels, &features)?;

            let predicted_index: usize = predicted_label.get_index();

            confusion_matrix[[actual_index, predicted_index]] += 1;
        }

        // Send intermediate confusion matrix from this thread to main thread
        transmitter.send(confusion_matrix).unwrap();
        println!("Thread {}\tFINISHED", thread_index);

        Ok(())
    }

    fn classify_helper<Num: ToPrimitive + Copy>
            (model: &Vec<GaussianFeature>, 
                labels: &Vec<ClassLabel>,
                sample_features: &Vec<Num>) 
            -> Result<Box<dyn Label>, ModelError> {
        let mut max_likelihood: f64 = 0.0;
        let mut best_label: Option<&ClassLabel> = None;

        // Combine the likelihoods of the occurrences of each feature
        for current_class in labels.iter() {        
            let feature_likelihoods: f64 = model.iter().enumerate()
                                            .fold(0.0, |total, (idx, feat)| {
                match feat.get_feature_likelihood_given_class(
                        sample_features[idx], current_class) {
                    // Use log rules and addition to avoid float underflow
                    Ok(prob) => total + prob.log10(),
                    Err(error) => panic!("{:?}", error)
                }
            });

            let class_likelihood: f64 = 
                model[0].get_class_likelihood(current_class)?;
            let likelihood: f64 = feature_likelihoods + class_likelihood.log10();

            if best_label.is_none() || likelihood > max_likelihood {
                max_likelihood = likelihood;
                best_label = Some(current_class);
            }
        }

        Ok(Box::new(*best_label.expect("Could not classify record.")))
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
            // split each line by space to get index and ascii values
            let components: Vec<String> = match line {
                Ok(s) => s.split(" ").map(|s| s.to_string()).collect(),
                Err(_) => panic!("Could not parse label.")
            };

            // Save parsed values in a ClassLabel struct
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
        self.add_values_from_file::<Num>(&file_path, 0)?;

        println!("Adding distribution standard deviations.");
        self.add_values_from_file::<Num>(&file_path, 1)?;

        for feature in self.features.iter_mut() {
            feature.prepare();
        }

        Ok(())
    }

    fn test<Num: ToPrimitive + Copy + FromStr>
            (&self, file_path: &String, multithreaded: bool) 
            -> Result<f64, Box<dyn Error>> {
        if multithreaded {
            self.test_multithreaded::<Num>(&file_path)
        } else {
            self.test_linear::<Num>(&file_path)
        }
    }

    fn classify<Num: ToPrimitive + Copy>
            (&self, sample_features: &Vec<Num>) 
            -> Result<Box<dyn Label>, ModelError> {
        GaussianNaiveBayes::classify_helper(
            &self.features, &self.labels, &sample_features)
    }
}
