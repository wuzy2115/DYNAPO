import os
import argparse
import subprocess
from glob import glob
import tempfile
import shutil

def find_image_folders(input_dir):
    """
    Traverse the directory structure to find subfolders containing images.
    """
    image_folders = []
    for root, dirs, _ in os.walk(input_dir):
        for dir in dirs:
            folder_path = os.path.join(root, dir, 'images')
            image_folders.append(folder_path)
    return image_folders

def create_batch(image_paths, batch_size=10):
    """
    Split image paths into batches of the given size.
    """
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    return batches

def run_colmap(input_images, output_path, colmap_exe_command, matching_type, vocab_tree_path=None):
    """
    Run COLMAP on the given batch of images.
    """
    # Prepare paths
    database_path = os.path.join(output_path, 'database.db')
    colmap_result_path = os.path.join(output_path, 'sparse')

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(colmap_result_path, exist_ok=True)

    # Create a temporary folder to store the batch of images
    with tempfile.TemporaryDirectory() as temp_image_folder:
        # Copy images to the temporary folder
        for image in input_images:
            shutil.copy(image, temp_image_folder)


        command = [colmap_exe_command, 'automatic_reconstructor', '--workspace_path', output_path, '--image_path', temp_image_folder, '--dense', '0']
        subprocess.run(command)

        # # Step 1: Feature extraction
        # command = [colmap_exe_command, 'feature_extractor', '--image_path', temp_image_folder, '--database_path', database_path]
        # subprocess.run(command)

        # # Step 2: Feature matching
        # command = [colmap_exe_command, matching_type, '--database_path', database_path]
        # if matching_type == 'vocab_tree_matcher':
        #     command += ['--VocabTreeMatching.vocab_tree_path', vocab_tree_path]
        # subprocess.run(command)

        # # Step 3: Mapping (Sparse Reconstruction)
        # command = [colmap_exe_command, 'mapper', '--database_path', database_path, '--image_path', temp_image_folder, '--output_path', colmap_result_path]
        # subprocess.run(command)

        # # Check if there are multiple sub-models in the result folder
        # submodels = [f for f in os.listdir(colmap_result_path) if os.path.isdir(os.path.join(colmap_result_path, f))]

        # # If there are multiple sub-models, merge them incrementally
        # if len(submodels) > 1:
        #     print(f"Multiple sub-models detected: {len(submodels)}. Merging them...")

        #     # Prepare paths for merging
        #     submodel_paths = [os.path.join(colmap_result_path, submodel) for submodel in submodels]
        #     merged_model_path = os.path.join(output_path, 'merged_model')
        #     current_merged_model = submodel_paths[0]  # Start with the first sub-model

        #     # Incrementally merge sub-models
        #     for submodel_path in submodel_paths[1:]:
        #         print(f"Merging {current_merged_model} with {submodel_path}...")
        #         # Run the model_merger command
        #         command = [colmap_exe_command, 'model_merger', '--input_path1', current_merged_model, '--input_path2', submodel_path, '--output_path', merged_model_path]
        #         os.makedirs(merged_model_path, exist_ok=True)
        #         subprocess.run(command)
        #         # Update the current merged model path for the next iteration
        #         current_merged_model = merged_model_path

        #     # Run bundle adjustment after merging all models
        #     refined_merged_model_path = os.path.join(output_path, 'refined_merged_model')
        #     command = [colmap_exe_command, 'bundle_adjuster', '--input_path', merged_model_path, '--output_path', refined_merged_model_path]
        #     subprocess.run(command)

        #     print(f"Models merged and refined. Results saved in: {refined_merged_model_path}")
        # else:
        #     print("No sub-models detected, or only one model was reconstructed.")
                
        print(f"COLMAP results saved in: {colmap_result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("colmap", add_help=True)
    parser.add_argument("--input_dir", "-i", type=str, help="path to image files")
    parser.add_argument("--output_dir", "-o", type=str, help="path to output files")
    parser.add_argument("--matcher", "-m", type=str, help="matching type", default='exhaustive_matcher')
    parser.add_argument("--batch_size", type=int, help='frames used for sparse reconstruction', default=10000)
    args = parser.parse_args()

    # ------------ Input settings -------------
    input_image_path = args.input_dir
    colmap_exe_command = 'colmap'  # Assuming colmap is available in the path
    matching_type = args.matcher  # 'vocab_tree_matcher' or 'exhaustive_matcher'

    if matching_type == 'vocab_tree_matcher':
        vocab_tree_path = 'vocab_tree_flickr100K_words1M.bin'
        if not os.path.exists(vocab_tree_path):
            os.system('wget https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin')
            #os.system('mv vocab_tree_flickr100K_words1M.bin ./weights/')
    else:
        vocab_tree_path = None

    # ------------ Find and process image folders -------------
    image_folders = find_image_folders(input_image_path)

    for folder in image_folders:
        print(f"Processing folder: {folder}")

        # image_paths = sorted(glob(os.path.join(folder, "*_0.jpg")))
        image_paths = sorted(glob(os.path.join(folder, "*.png")))
        output_path = os.path.join(args.output_dir, folder.split('/')[-2])
        run_colmap(image_paths, output_path, colmap_exe_command, matching_type, vocab_tree_path)
