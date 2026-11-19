# BIRA · Segmentation FoodSeg103 sur Jetson
Ce modèle DINOv2 reconnaît 103 catégories alimentaires et produit un masque où chaque pixel porte un identifiant de classe. Le modèle ONNX comprend l’encodeur et la tête de segmentation entraînée par BIRA. Les noms des classes sont dans `id2label.json` (0 = arrière-plan).

## 1. Récupérer le modèle sur la Jetson
Depuis un terminal de la Jetson Orin NX 16 Go, avec Git LFS installé :
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch feature/ensemble-dinov2-4694 --single-branch https://github.com/Eric-mmmm/BIRA-foodsegCV.git BIRA-dinov2
cd BIRA-dinov2
git lfs pull --include="src/foodv/models/ensemble/foodseg103.onnx"
cd src/foodv/models/ensemble
sha256sum -c MODEL.sha256
```
Attendre `foodseg103.onnx: OK`. Après fusion de la PR, ces instructions fonctionnent aussi avec la branche `main`.

## 2. Construire le moteur TensorRT une fois sur la carte
Les commandes ci-dessous ciblent TensorRT 10 avec JetPack. Rester dans ce dossier. Relever le mode de puissance actif pour pouvoir comparer deux essais.
```bash
cat /etc/nv_tegra_release
nvpmodel -q
TRT="$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)"
"$TRT" --version
set -o pipefail
"$TRT" --onnx=foodseg103.onnx --fp16 --saveEngine=foodseg103_fp16.engine --skipInference 2>&1 | tee build.log
```
Continuer après une construction réussie. Reconstruire le moteur sur la carte si la version de TensorRT change. Le profil fixe attend une image RGB de 448 × 448, batch 1.

## 3. Mesurer le moteur seul
```bash
"$TRT" --loadEngine=foodseg103_fp16.engine --infStreams=1 --noDataTransfers --warmUp=2000 --duration=30 2>&1 | tee bench_engine.log
```
Dans la ligne `GPU Compute Time`, relever `median` et `percentile(95%)` en millisecondes. Ce test chronomètre le calcul du réseau sur des entrées synthétiques, transferts exclus. Le débit affiché par trtexec peut inclure du pipelining.

## 4. Mesurer une vraie photo jusqu’au masque
Utiliser un environnement Python Jetson avec TensorRT 10, PyTorch CUDA compatible avec JetPack, NumPy et Pillow ≥ 9.1. Vérifier les imports puis remplacer le chemin de la photo :
```bash
python3 -c "import tensorrt, torch, numpy, PIL; print(tensorrt.__version__, torch.__version__); assert torch.cuda.is_available()"
python3 bench_jetson.py /chemin/assiette.jpg --warmup 10 --runs 100 --mask assiette_ids.png | tee bench_photo.json
```
Le script charge une fois la photo et le moteur, puis mesure redimensionnement, normalisation, transferts, inférence, sélection de classe par pixel et remise du masque aux dimensions de la photo. La copie des résultats vers le CPU attend la fin du calcul GPU. La lecture du fichier, la capture caméra et l’écriture du PNG sont exclues. Relever `photo_to_mask_p50_ms` (temps typique) et `photo_to_mask_p95_ms` (95 % des essais terminés dans ce temps). Le PNG conserve les identifiants bruts. Pour comparer deux configurations, garder la même photo, sa résolution et le mode de puissance. Conserver les deux logs avec la version de JetPack.

## Référence mesurée sur PC
Sur les 2 135 images du benchmark FoodSeg103, l’ONNX obtient **46,94 % de mIoU** (moyenne des 103 classes alimentaires, arrière-plan exclu). Sur RTX 4060 Laptop avec ONNX Runtime CUDA : médiane **41,38 ms** pour le réseau et **46,86 ms** de la photo décodée au masque. Les mesures TensorRT FP16 sur Jetson restent à faire. Pour comparer les masques, `python predict_photo.py assiette.jpg reference_ids.png` produit la référence ONNX dans un environnement ONNX Runtime CUDA. Les résultats détaillés figurent dans [resultats.json](resultats.json) et [classes.csv](classes.csv).

Encodeur initial : [DINOv2 ViT-B/14](https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m/tree/4685c99dabffe5affac90bd99dbffd25801ae58d). Tête entraînée sur 4 485 photos et sélectionnée sur 498 autres photos du split d’entraînement.
