from falcor import *
import os


repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESTIRPT_SCENE = os.path.join(repo_dir, "media", "test_scenes", "cornell_box.pyscene")
RESTIRPT_OUTPUT_DIR = os.path.join(repo_dir, "outputs", "restirpt_smoke", "hybrid_fixed_x2")
RESTIRPT_INTEGRATOR = "restirpt"
RESTIRPT_WIDTH = 320
RESTIRPT_HEIGHT = 180
RESTIRPT_FRAMES = 0
RESTIRPT_SAMPLES_PER_PIXEL = 1
RESTIRPT_INITIAL_CANDIDATE_COUNT = 1
RESTIRPT_ACCUMULATE = False
RESTIRPT_CAPTURE_LABEL = "ReSTIRPT_hybrid_fixed_x2_smoke"
RESTIRPT_VBUFFER_SAMPLES = 1
RESTIRPT_USE_SPATIAL_REUSE = True
RESTIRPT_SPATIAL_NEIGHBOR_COUNT = 3
RESTIRPT_SPATIAL_RADIUS = 20
RESTIRPT_SPATIAL_ITERATIONS = 1

# ShiftMapping::Hybrid, HybridShiftMode::FixedX2.
# This fast shader-load smoke test avoids the prepared-prefix retrace path by
# selecting the fixed-x2 hybrid mode explicitly.
RESTIRPT_SHIFT_MAPPING = 2
RESTIRPT_HYBRID_SHIFT_MODE = 0

m.script(os.path.join(repo_dir, "scripts", "RunReSTIRPTBatch.py"))
