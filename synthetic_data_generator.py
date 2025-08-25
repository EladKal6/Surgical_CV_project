#
# Computer Vision, Surgical Applications - Project Phase 1
# Final Version - Adds randomized camera distance for scale variation.
#
import bpy
import os
import random
import math
import json
import sys

# --- 1. CONFIGURATION ---
BASE_DIR = "/datashare/project"
BASE_MODELS_DIR = os.path.join(BASE_DIR, "surgical_tools_models")
BACKGROUNDS_DIR = os.path.join(BASE_DIR, "train2017")
HDRIS_DIR = os.path.join(BASE_DIR, "haven", "hdris")
OUTPUT_DIR = "/home/student/synthetic_output"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

TOOL_CONFIG = {"tweezers": {"folder": "tweezers"}}

RENDER_WIDTH = 960
RENDER_HEIGHT = 540


# --- 2. UTILITY FUNCTIONS ---

def setup_scene():
    bpy.ops.object.select_all(action='SELECT');
    bpy.ops.object.delete()

    # === THIS IS THE CRITICAL CHANGE ===
    # The camera is now placed at a RANDOM distance (Z-axis) for each image.
    # The range (2.0 to 4.0) will produce a good variety of scales.
    camera_z_distance = random.uniform(9.0, 14.0)
    print("camera_z_distance: ", camera_z_distance)
    bpy.ops.object.camera_add(location=(0, 0, camera_z_distance));
    bpy.context.scene.camera = bpy.context.object
    scene = bpy.context.scene;
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = RENDER_WIDTH;
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = 'RGBA'


def list_files_recursive(path, extensions):
    found_files = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith(tuple(extensions)): found_files.append(os.path.join(root, name))
    return found_files


def list_files_flat(path, extensions):
    try:
        return [f for f in os.listdir(path) if f.lower().endswith(tuple(extensions))]
    except Exception:
        return []


# --- 3. MAIN EXECUTION SCRIPT ---

def main(start_index, end_index):
    output_json_path = os.path.join(OUTPUT_DIR, f"annotations_{start_index}_to_{end_index - 1}.json")
    print(f"--- Starting Batch Generation: Images {start_index} to {end_index - 1} ---")
    print(f"--- Annotations will be saved to: {output_json_path} ---")

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
    all_images_annotations = []
    background_files = list_files_flat(BACKGROUNDS_DIR, ['.jpg', '.jpeg', '.png'])
    hdri_files = list_files_recursive(HDRIS_DIR, ['.hdr', '.exr'])
    if not background_files or not hdri_files: return

    mask_mat = bpy.data.materials.new(name="MaskMaterial")
    mask_mat.use_nodes = True
    bsdf = mask_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (1, 1, 1, 1);
    bsdf.inputs['Emission'].default_value = (1, 1, 1, 1)

    for i in range(start_index, end_index):
        print(f"--- Generating sample {i + 1} ---")
        tool_type = random.choice(list(TOOL_CONFIG.keys()))
        tool_folder_path = os.path.join(BASE_MODELS_DIR, TOOL_CONFIG[tool_type]["folder"])
        model_files = list_files_flat(tool_folder_path, ['.obj'])
        if not model_files: continue

        setup_scene()  # This now creates a camera at a random distance

        model_path = os.path.join(tool_folder_path, random.choice(model_files))
        bpy.ops.import_scene.obj(filepath=model_path)
        instrument_obj = bpy.context.selected_objects[0]
        x_shift = random.uniform(-1.5, 1.5)
        y_shift = random.uniform(-1.5, 1.5)
        z_shift = random.uniform(-0.2, 0.2)
        instrument_obj.location = (x_shift, y_shift, z_shift)
        instrument_obj.rotation_euler = (
            math.radians(random.uniform(-180, 180)), math.radians(random.uniform(-180, 180)),
            math.radians(random.uniform(-180, 180)))
        bpy.context.scene.camera.location.x += random.uniform(-0.1, 0.1)
        bpy.context.scene.camera.location.y += random.uniform(-0.1, 0.1)

        # Render the Color Image
        world = bpy.context.scene.world;
        if world is None: world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world;
        world.use_nodes = True
        world_nodes = world.node_tree.nodes;
        world_nodes.clear()
        env_tex = world_nodes.new(type='ShaderNodeTexEnvironment');
        env_tex.image = bpy.data.images.load(random.choice(hdri_files))
        bg_node = world_nodes.new(type='ShaderNodeBackground');
        output_node = world_nodes.new(type='ShaderNodeOutputWorld')
        world.node_tree.links.new(env_tex.outputs['Color'], bg_node.inputs['Color'])
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

        bpy.context.scene.use_nodes = True;
        tree = bpy.context.scene.node_tree;
        tree.nodes.clear()
        rl = tree.nodes.new('CompositorNodeRLayers');
        alpha_over = tree.nodes.new('CompositorNodeAlphaOver')
        img_node = tree.nodes.new('CompositorNodeImage');
        img_node.image = bpy.data.images.load(os.path.join(BACKGROUNDS_DIR, random.choice(background_files)))
        scale_node = tree.nodes.new('CompositorNodeScale');
        scale_node.space = 'RENDER_SIZE'
        composite_node = tree.nodes.new('CompositorNodeComposite')
        tree.links.new(img_node.outputs['Image'], scale_node.inputs['Image'])
        tree.links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
        tree.links.new(rl.outputs['Image'], alpha_over.inputs[2])
        tree.links.new(alpha_over.outputs['Image'], composite_node.inputs['Image'])

        mat = bpy.data.materials.new(name="RandomMetallic")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs['Metallic'].default_value = 1.0;
        bsdf.inputs['Roughness'].default_value = random.uniform(0.1, 0.5)
        if instrument_obj.data.materials:
            instrument_obj.data.materials[0] = mat
        else:
            instrument_obj.data.materials.append(mat)

        img_filename = f"synth_{i:04d}_{tool_type}.png"
        img_filepath = os.path.join(OUTPUT_IMAGES_DIR, img_filename)
        bpy.context.scene.render.filepath = img_filepath
        bpy.ops.render.render(write_still=True)

        # Render the Mask Image
        bpy.context.scene.use_nodes = False
        world.use_nodes = False
        if instrument_obj.data.materials:
            instrument_obj.data.materials[0] = mask_mat
        else:
            instrument_obj.data.materials.append(mask_mat)

        mask_filename = f"synth_{i:04d}_{tool_type}_mask.png"
        mask_filepath = os.path.join(OUTPUT_MASKS_DIR, mask_filename)
        bpy.context.scene.render.filepath = mask_filepath
        bpy.ops.render.render(write_still=True)

        annotation = {"image_path": img_filepath, "mask_path": mask_filepath, "tool_type": tool_type}
        all_images_annotations.append(annotation)

    print(f"--- Saving annotations to {output_json_path} ---")
    with open(output_json_path, 'w') as f:
        json.dump({"annotations": all_images_annotations}, f, indent=4)
    print("--- Script finished successfully. ---")


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
        if len(argv) >= 2:
            # Optionally accept BASE_DIR and OUTPUT_DIR as arguments
            if len(argv) >= 4:
                BASE_DIR = argv[2]
                OUTPUT_DIR = argv[3]
                OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
                OUTPUT_MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
                # Update dependent directories
                BASE_MODELS_DIR = os.path.join(BASE_DIR, "surgical_tools_models")
                BACKGROUNDS_DIR = os.path.join(BASE_DIR, "train2017")
                HDRIS_DIR = os.path.join(BASE_DIR, "haven", "hdris")
    if len(argv) == 2:
        try:
            start_index = int(argv[0])
            end_index = int(argv[1])
            main(start_index, end_index)
        except ValueError:
            print("Error: Invalid arguments. Please provide integers for start and end indices.")
    else:
        print("Usage: blender --background --python script.py -- <start_index> <end_index>")
        print("Example: blender --background --python script.py -- 0 600")
        print("Running a default small test batch (0 to 5).")
        main(0, 5)