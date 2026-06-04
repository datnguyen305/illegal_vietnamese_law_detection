# illegal_vietnamese_law_detection

This project is a single-image Vietnamese traffic violation detection pipeline.
It focuses on three violations:

- illegal parking
- wrong lane
- crossing the red light

The goal is not to make the object detector directly decide the violation. Instead,
the detector extracts visible evidence from the image, a geometry/rule layer turns
that evidence into structured facts, and a VLM can use those facts to produce a
final explanation for RAG.

## Pipeline Overview

```text
Input image
  -> object detection
  -> spatial fact extraction
  -> rule-based violation candidates
  -> violated-object segmentation assets
  -> VLM-ready reasoning prompt
  -> RAG query for Vietnamese traffic law
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

The local VLM path uses Qwen2.5-VL through Hugging Face Transformers. The first run
will download model weights, so it needs internet access and enough disk space.

## What Each Stage Does

### 1. Input Image

The input is one traffic image, not a video. Because of that, the system can reason
about object positions, but it cannot fully prove movement or parking duration.

Example:

```bash
python main.py dataset/image/parking/violate/2.png
```

### 2. Object Detection

The detector currently uses GroundingDINO through the `Object_detector` class.
It searches for traffic-related objects such as:

- cars, buses, trucks, motorbikes, bicycles
- red/yellow/green traffic lights
- stop lines
- lane markings
- crosswalks
- sidewalks
- no-parking signs
- road regions

Detection output is normalized into JSON records like:

```json
{
  "label": "motorbike",
  "score": 0.91,
  "bbox": [300.0, 500.0, 420.0, 700.0]
}
```

### 3. Scene Geometry

For single-image traffic reasoning, object boxes alone are usually not enough.
The system can optionally use a scene config file with fixed-camera geometry:

- lane polygons
- allowed vehicle types per lane
- no-parking zones
- stop-line position

See:

```text
configs/scene_config.example.json
```

Example lane config:

```json
{
  "id": "motorbike_lane",
  "allowed_vehicle_types": ["motorbike", "bicycle"],
  "polygon": [[0, 420], [260, 400], [300, 720], [0, 720]]
}
```

The pipeline checks each vehicle's bottom-center point against these polygons.
This produces facts such as:

```json
{
  "label": "motorbike",
  "bottom_center": [360.0, 700.0],
  "lane": {
    "id": "car_lane",
    "allowed_vehicle_types": ["car", "bus", "truck"]
  },
  "inside_no_parking_zone": false,
  "past_stop_line": true
}
```

### 4. Rule-Based Violation Candidates

The rule layer creates candidate violations from the structured facts.

Wrong lane:

```text
if vehicle is inside a lane and vehicle type is not allowed in that lane:
    candidate = wrong_lane
```

Crossing the red light:

```text
if red traffic light is detected and vehicle is past the stop line:
    candidate = crossing_red_light
```

Illegal parking:

```text
if vehicle is inside a configured no-parking zone:
    candidate = illegal_parking
```

For illegal parking and red-light crossing, the pipeline uses `suspected` status
because a single image cannot prove time-based behavior.

### 5. Violated-Object Segmentation

After the rule engine creates violation candidates, the pipeline can export visual
assets for the vehicles involved in those candidates.

Run with:

```bash
python main.py dataset/image/parking/violate/2.png \
  --scene-config configs/scene_config.example.json \
  --segment-dir outputs/segments
```

For each violated vehicle, the pipeline saves:

- a crop of the vehicle
- a binary mask image
- an overlay image showing the violated object on the original image

The current implementation uses the vehicle bounding box as the mask:

```text
method = bbox_mask
```

This is useful for reports and VLM evidence because it isolates the object that
triggered the violation candidate. It is not yet pixel-accurate semantic segmentation.
For better masks, this stage can be replaced with SAM/SAM2 using the detected vehicle
bounding box as the segmentation prompt.

Example segment output:

```json
{
  "violation_type": "wrong_lane",
  "vehicle_id": "vehicle_0",
  "vehicle_label": "motorbike",
  "method": "bbox_mask",
  "crop_path": "outputs/segments/000_wrong_lane_vehicle_0_crop.png",
  "mask_path": "outputs/segments/000_wrong_lane_vehicle_0_mask.png",
  "overlay_path": "outputs/segments/000_wrong_lane_vehicle_0_overlay.png"
}
```

### 6. Qwen2.5-VL Reasoning

The pipeline generates a prompt and can optionally send it to Qwen2.5-VL.

Qwen receives:

- the original image
- detector facts
- rule-engine candidates
- instruction to return strict JSON

The VLM should classify exactly one of:

- `illegal_parking`
- `wrong_lane`
- `crossing_red_light`
- `no_violation_or_insufficient_evidence`

The VLM is used for explanation and final reasoning, not raw object detection.
The generated prompt is always saved in `vlm_prompt`. When Qwen is enabled, the
model response is saved in `vlm_result`.

### 7. RAG Query

The pipeline also generates legal retrieval queries. These can be used by a RAG
system to fetch relevant Vietnamese traffic law or fine information.

Example:

```json
[
  "Vietnam traffic law wrong lane motorbike car fine",
  "Vietnam traffic law crossing red light fine"
]
```

## Run Image Analysis

Basic run:

```bash
python main.py dataset/image/parking/violate/2.png
```

Save JSON and an annotated image:

```bash
python main.py dataset/image/parking/violate/2.png \
  --output outputs/parking_2.json \
  --annotate outputs/parking_2.png \
  --segment-dir outputs/parking_2_segments
```

Use fixed-camera geometry for stronger single-image reasoning:

```bash
python main.py dataset/image/traffic_light/violate/have_traffic_light/2.png \
  --scene-config configs/scene_config.example.json \
  --output outputs/traffic_light_2.json
```

Run Qwen2.5-VL after detection and rules:

```bash
python main.py dataset/image/parking/violate/2.png \
  --scene-config configs/scene_config.example.json \
  --segment-dir outputs/parking_2_segments \
  --output outputs/parking_2_qwen.json \
  --vlm qwen
```

Use a different Qwen2.5-VL size:

```bash
python main.py dataset/image/parking/violate/2.png \
  --scene-config configs/scene_config.example.json \
  --vlm qwen \
  --qwen-model Qwen/Qwen2.5-VL-7B-Instruct
```

CLI options:

```text
--scene-config      Optional geometry JSON file
--annotate          Save image with detected bounding boxes
--segment-dir       Save violated-object crops, masks, and overlays
--output            Save JSON result
--threshold         Object detection box threshold
--text-threshold    Object detection text threshold
--vlm               Use no VLM or Qwen2.5-VL: none, qwen
--qwen-model        Hugging Face model id for Qwen2.5-VL
--vlm-max-new-tokens Maximum number of tokens generated by the VLM
```

## Output

The CLI returns JSON with:

- `detections`: detected vehicles, traffic lights, stop lines, signs, lanes, and road objects
- `facts`: normalized vehicle positions and scene state
- `violations`: rule-engine candidates
- `violated_object_segments`: saved crop/mask/overlay paths for candidate violating vehicles
- `vlm_prompt`: prompt sent to Qwen2.5-VL or saved for inspection
- `vlm_result`: Qwen2.5-VL response when `--vlm qwen` is used
- `rag_queries`: search queries for Vietnamese traffic law retrieval

Example output structure:

```json
{
  "image_path": "dataset/image/parking/violate/2.png",
  "image_size": {
    "width": 960,
    "height": 720
  },
  "detections": [],
  "facts": {
    "vehicles": [],
    "traffic_light_state": "unknown",
    "has_stop_line": false,
    "has_no_parking_sign": false,
    "has_scene_config": false
  },
  "violations": [],
  "violated_object_segments": [],
  "vlm_prompt": "...",
  "vlm_result": null,
  "rag_queries": []
}
```

## RAG Application Setup

The RAG starter code lives in:

```text
rag/
```

It consumes the JSON produced by `main.py`, especially:

- `violations`
- `rag_queries`
- `vlm_result`

Run a basic RAG context build:

```bash
python rag_demo.py outputs/result.json \
  --output outputs/rag_context.json
```

The RAG output contains:

- `primary_violation`
- `retrieved_documents`
- `context`
- `final_answer_prompt`

Important files:

```text
rag/knowledge_base.json
```

Placeholder legal knowledge base. Replace this with official Vietnamese traffic
law excerpts, article/decree references, fine ranges, and source URLs.

```text
rag/keyword_retriever.py
```

Simple baseline retriever. Your friend can replace it with FAISS, Chroma, Qdrant,
BM25, or another vector search backend.

```text
rag/context_builder.py
```

Builds the final legal context and LLM prompt from the detection output.

See [rag/README.md](rag/README.md) for the handoff details.

## Main Files

```text
main.py
```

Command-line entry point.

```text
illegal_detector/traffic_violation_pipeline.py
```

Main pipeline. It runs detection, creates facts, applies rules, builds the VLM
prompt, and creates RAG queries.

```text
illegal_detector/object_detector/model_oop.py
```

GroundingDINO object detector wrapper.

```text
illegal_detector/violated_object_segmenter.py
```

Exports crops, bbox masks, and overlays for vehicles involved in violation candidates.

```text
illegal_detector/vlm_reasoner.py
```

Runs Qwen2.5-VL on the original image plus the structured prompt.

```text
rag_demo.py
```

Command-line entry point for the RAG context builder.

```text
configs/scene_config.example.json
```

Example geometry file for fixed-camera lane, no-parking-zone, and stop-line rules.

## Single-Image Limits

Wrong-lane detection is the best fit for a single image when lane polygons are available.
Crossing a red light can be marked as suspected when a red light and a vehicle past the
stop line are visible. Illegal parking should be treated as suspected unless the dataset
label or external evidence proves parking duration.

In a real deployment, video tracking would make illegal parking and red-light crossing
more reliable. For this course project, the pipeline is designed to make single-image
reasoning explicit and defensible.
