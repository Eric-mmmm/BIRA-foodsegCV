# Ensemble ONNX pour Jetson
B1 vérifie la présence de nourriture, EfficientNet ou ConvNeXt nomme le plat et DeepLab produit le masque alimentaire.

Depuis un clone de cette branche avec Git LFS installé :
```bash
git lfs pull --include="src/foodcv/models/ensemble/*.onnx"
cd src/foodcv/models/ensemble
sha256sum -c MODELS.sha256
bash build.sh convnext
```

Avec TensorRT 10, les moteurs se construisent sur la Jetson. Le test photo se lance avec `python3 bench_photo.py assiette.jpg`. Les mesures dans pc_results.json proviennent de la RTX 4060 Laptop. La mesure sur Jetson reste à faire.

[Archive complète et guide](https://github.com/Eric-mmmm/BIRA-foodsegCV/releases/tag/jetson-onnx-2026-11-19)
