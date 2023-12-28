import torch
from model import PoseEstimationWithMobileNet
from load_state import load_state

def convert_to_onnx(net, output_name):
    input = torch.randn(1, 3, 256, 456)
    input_names = ['data']
    output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                    'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    torch.onnx.export(net, input, output_name, verbose=True, input_names=input_names, output_names=output_names)


# Specify the path to the checkpoint file
checkpoint_path = "checkpoint_iter_370000.pth"

# Create an instance of the PoseEstimationWithMobileNet model
model = PoseEstimationWithMobileNet(num_refinement_stages=1)

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

load_state(model, checkpoint)
convert_to_onnx(model, "pose_estimator.onnx")