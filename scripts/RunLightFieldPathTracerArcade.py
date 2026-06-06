from falcor import *
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
output_dir = globals().get("LF_OUTPUT_DIR", os.path.join(repo_dir, "outputs", "lightfield"))
scene_path = globals().get("LF_SCENE", "Arcade/Arcade.pyscene")
view_count_x = globals().get("LF_VIEW_COUNT_X", 8)
view_count_y = globals().get("LF_VIEW_COUNT_Y", 1)
view_width = globals().get("LF_VIEW_WIDTH", 512)
view_height = globals().get("LF_VIEW_HEIGHT", 512)
baseline_x = globals().get("LF_BASELINE_X", 0.08)
baseline_y = globals().get("LF_BASELINE_Y", 0.0)
frames_per_view = globals().get("LF_FRAMES_PER_VIEW", 8)
samples_per_pixel = globals().get("LF_SAMPLES_PER_PIXEL", 1)

m.script(os.path.join(script_dir, "LightFieldPathTracer.py"))
m.loadScene(scene_path)
m.resizeFrameBuffer(view_width, view_height)

print(
    "Light-field batch: {}x{} views, {}x{} pixels, {} frames/view, {} samples/pixel".format(
        view_count_x, view_count_y, view_width, view_height, frames_per_view, samples_per_pixel
    )
)

render_light_field_views(
    m,
    view_count_x=view_count_x,
    view_count_y=view_count_y,
    baseline_x=baseline_x,
    baseline_y=baseline_y,
    frames_per_view=frames_per_view,
    output_dir=output_dir,
)

exit()
