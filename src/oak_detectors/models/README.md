# Model Layout

Place detector model assets under this package so runtime paths stay local to `oak_detectors`.

Layout:

```text
models/
├── money/         best.rvc2.tar.xz — DepthAI NN Archive (banknote model, mode 2)
├── myfriend_face/ best.rvc2.tar.xz — DepthAI NN Archive (friend faces, mode 3; classes = names)
└── navigation/    empty — walk mode is stereo-depth-only, no model needed
```

`money/` and `myfriend_face/` use the same NN Archive (YOLO RVC2) format and the same
`DetectionNetwork` pipeline in `main_pipeline.py`; they differ only in model, labels, and
output topic. The face model's classes ARE the friend names — the system just speaks
whichever name it detects.

`navigation/` is kept (with `.gitkeep`) only as a placeholder; walk-mode obstacle
detection runs entirely on stereo depth (see HANDOFF "Walk pipeline details").
