use std::{io, path::Path};

use crate::{
    data::{MnistBatch, MnistBatcher},
    model::{Model, ModelConfig},
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    prelude::*,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

impl TrainStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl InferenceStep for Model {
    type Input = MnistBatch;
    type Output = ClassificationOutput;

    fn step(&self, batch: MnistBatch) -> ClassificationOutput {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn prepare_artifact_dir(artifact_dir: &str) -> io::Result<()> {
    let experiment_log = Path::new(artifact_dir).join("experiment.log");
    match std::fs::remove_file(experiment_log) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    std::fs::create_dir_all(artifact_dir)
}

pub fn train(
    artifact_dir: &str,
    config: TrainingConfig,
    device: impl Into<Device>,
) -> Result<(), Box<dyn std::error::Error>> {
    prepare_artifact_dir(artifact_dir)?;
    config.save(format!("{artifact_dir}/config.json"))?;

    let device = device.into();
    device.seed(config.seed);
    let autodiff_device = device.autodiff();

    let batcher = MnistBatcher::default();
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());
    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_default_checkpointers()
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init(&autodiff_device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .into_record()
        .save(format!("{artifact_dir}/model"))?;

    Ok(())
}
