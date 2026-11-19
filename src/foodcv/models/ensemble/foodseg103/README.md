# Segmenter les 103 catégories alimentaires

Ce profil utilise DeepLabV3+ avec un encodeur ResNet50 entraîné par mawiie sur FoodSeg103. Chaque pixel reçoit un identifiant de 0 à 103. Le fichier [id2label.json](id2label.json) associe ces identifiants aux noms du jeu de données. Le fond vaut 0 et « other ingredients » vaut 103. La classe 48 réunit poulet et canard. La sortie décrit plusieurs ingrédients dans une même assiette.

## Utilisation sur Jetson

Suivre les prérequis TensorRT 10 et PyTorch CUDA du [guide ensemble](../README.md). Depuis la racine du dépôt sur la branche contenant ce profil :

```bash
git lfs pull --include="src/foodcv/models/ensemble/foodseg103/*.onnx"
cd src/foodcv/models/ensemble
(cd foodseg103 && sha256sum -c MODEL.sha256)
mkdir -p engines
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
"$TRTEXEC" --onnx=foodseg103/foodseg103.onnx --fp16 --saveEngine=engines/foodseg103.plan
python3 foodseg103/segment.py ~/assiette.jpg --output assiette-ingredients.png
```

Le PNG contient les identifiants de classe aux dimensions de la photo. La console affiche les noms prédits et leur nombre de pixels. De petites régions peuvent être des erreurs. Le réseau traite une image RGB redimensionnée à 256 × 256 et normalisée avec les moyennes et écarts types ImageNet. Le script agrandit la carte de classes au plus proche voisin.

## Origine et vérification

Poids issus du [checkpoint PyTorch publié](https://huggingface.co/mawiie/foodseg103-deeplabv3plus-resnet50/tree/3933ade0ff254d236b993917eedf89d0a4c3d0e8), dont la fiche indique MIT. Identifiants issus de la [table FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103/blob/34e1208e14bc3595d544fc8c3f3c6673253fd9ef/id2label.json). Export ONNX FP32, opset 17. Sur la première photo de validation du jeu Hugging Face, les sorties PyTorch et ONNX Runtime CUDA avaient la forme 1 × 104 × 256 × 256. Les classes prédites concordaient sur les 65 536 pixels. Cette vérification porte sur la conversion. La précision par catégorie et les mesures TensorRT sur la Jetson BIRA restent à établir.
