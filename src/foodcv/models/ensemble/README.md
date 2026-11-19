# Tester la vision alimentaire de BIRA sur une Jetson Orin NX 16 Go

À partir d’une photo d’assiette, ce test renvoie un score de présence de nourriture, un nom de plat et une image où chaque pixel porte le numéro d’une catégorie alimentaire. Il utilise trois modèles à la fois. B1 évalue la présence de nourriture, EfficientNet nomme le plat et DeepLab trace les régions. ConvNeXt peut remplacer EfficientNet pour comparer les deux classifieurs sur la même photo.

## 1. Ouvrir un terminal sur la Jetson

Avec un écran et un clavier branchés à la Jetson, ouvrir le terminal Ubuntu. Pour utiliser le PC à la place, relever le compte et l’adresse de la carte dans ce terminal avec `whoami` et `hostname -I`. Puis ouvrir PowerShell sur le PC et exécuter la commande suivante en remplaçant les deux valeurs. Avec la liaison USB de BIRA, l’adresse observée était `192.168.55.1`.

```powershell
ssh utilisateur@adresse-de-la-jetson
```

Toutes les commandes suivantes s’exécutent **sur la Jetson**, sauf la copie de la photo explicitement indiquée à l’étape 5. Pour télécharger depuis GitHub, connecter la carte à Internet par Ethernet ou Wi-Fi.

## 2. Vérifier les outils et le GPU

```bash
cat /etc/nv_tegra_release
df -h ~
free -h
nvpmodel -q
command -v trtexec || ls /usr/src/tensorrt/bin/trtexec
python3 -c "import tensorrt as t, torch, torchvision, numpy, PIL; print('TensorRT', t.__version__, 'PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
```

Le script photo utilise les API de **TensorRT 10** et attend **CUDA True**. Si un import échoue, activer l’environnement Python de la carte qui contient ces bibliothèques. Si elles manquent, relever la version de JetPack avant d’installer TensorRT, PyTorch avec CUDA et torchvision compatibles. Pour compiler les moteurs à l’étape 4, seul l’outil TensorRT `trtexec` est nécessaire. [Installation NVIDIA de PyTorch pour Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

## 3. Télécharger et vérifier les quatre modèles

Sur une nouvelle installation, installer Git LFS puis cloner le dépôt. Si le dépôt existe déjà, utiliser le clone de l’équipe à jour sur `main` et passer directement au deuxième bloc.

```bash
sudo apt update && sudo apt install -y git git-lfs
git lfs install
cd ~
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Eric-mmmm/BIRA-foodsegCV.git
```

```bash
cd ~/BIRA-foodsegCV
git lfs pull --include="src/foodcv/models/ensemble/*.onnx"
cd src/foodcv/models/ensemble
sha256sum -c MODELS.sha256
```

Attendre **quatre lignes OK**, une pour chaque fichier ONNX. Une erreur de somme ou un fichier ONNX de quelques centaines d’octets indique un téléchargement incomplet ou un pointeur Git LFS. Relancer `git lfs pull` à la racine du dépôt puis vérifier de nouveau. Si Git LFS est indisponible, la [release Jetson](https://github.com/Eric-mmmm/BIRA-foodsegCV/releases/tag/jetson-onnx-2026-11-19) fournit une archive et les commandes de téléchargement et de vérification.

## 4. Construire les moteurs TensorRT sur la carte

Depuis le dossier contenant ce README :

```bash
bash build.sh convnext
ls -lh engines/*.plan
```

Le script compile les quatre graphes avec FP16 autorisé et teste chaque moteur séparément. À la fin, vérifier la présence de `b1.plan`, `efficientnet.plan`, `deeplab.plan` et `convnext.plan`. Les journaux individuels se trouvent dans `logs/`. Pour construire uniquement le trio par défaut, utiliser `bash build.sh`. Recompiler les moteurs sur la carte après un changement de matériel ou de version TensorRT.

## 5. Tester une photo

Placer une photo dans le dossier personnel de la Jetson sous le nom `assiette.jpg`. Pour la copier depuis le PC, ouvrir **un deuxième PowerShell sur Windows**, dans le dossier de la photo :

```powershell
scp .\assiette.jpg utilisateur@adresse-de-la-jetson:~/assiette.jpg
```

Revenir au **terminal Jetson**, dans le dossier `ensemble`, puis lancer :

```bash
python3 bench_photo.py ~/assiette.jpg --runs 100
mv classes.png classes-efficientnet.png
python3 bench_photo.py ~/assiette.jpg --dish convnext --runs 100
mv classes.png classes-convnext.png
```

Chaque commande Python affiche un résultat JSON. `food_score` est le score produit par B1 et `food_accepted` indique s’il atteint son seuil. `dish` est le plat proposé et `dish_accepted` indique si son score atteint le seuil prévu. Le benchmark exécute les trois modèles sur chaque photo, même quand un seuil entraîne un rejet.

Le PNG a les dimensions de la photo. Ses pixels valent 0 pour le fond et 1 à 8 pour les aliments, dans l’ordre de [deeplab.json](deeplab.json). Il peut sembler presque noir à l’écran parce que ces valeurs sont faibles. Pour visualiser les contours de toutes les régions retenues sous forme de silhouette blanche :

```bash
python3 -c "from PIL import Image; Image.open('classes-efficientnet.png').point(lambda v: 255 if v else 0).save('silhouette.png')"
```

## 6. Mesurer la vitesse et la mémoire

Ouvrir un deuxième terminal **sur la Jetson** et démarrer la mesure avant de relancer le test photo dans le premier. Arrêter la mesure avec Ctrl+C après la fin du test.

```bash
cd ~/BIRA-foodsegCV/src/foodcv/models/ensemble
tegrastats --interval 200 --logfile logs/memoire.log
```

Conserver le JSON du test, le mode de puissance de l’étape 2 et le journal mémoire. Le script écarte cinq appels de chauffe. Sur les 100 appels suivants, `p50_ms` donne le délai médian et `p95_ms` le délai sous lequel finissent 95 % des appels. Le chronomètre va du redimensionnement d’une photo déjà décodée jusqu’au masque filtré en mémoire. La capture caméra, l’écriture du PNG et le mouvement du bras sont hors mesure.

Pour un objectif de 10 images/s, le budget par image est de 100 ms. Pour 30 images/s, il est d’environ 33,3 ms. Pour valider un objectif en direct, mesurer aussi le délai entre l’acquisition caméra et la disponibilité du masque avec les autres services du robot actifs. Le fichier [pc_results.json](pc_results.json) contient les mesures individuelles sur RTX 4060 Laptop. Les mesures sur la Jetson sont à produire avec les étapes ci-dessus.
