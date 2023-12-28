use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
fn main() -> ort::Result<()> {
    // Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
    ort::init()
        .with_name("Pose Estimation")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let _model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(16)?
        .with_model_from_file("/home/aboussejra/fun/rust_pose_estimate/python/pose_estimator.onnx")?;
    Ok(())
}
