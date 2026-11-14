# Guide complet du projet DonkeyCar — comprendre et refaire tout le projet

Ce guide explique **tout** le projet : les outils, les idées, puis chaque fichier de
code dans l'ordre où il faut le lire, et enfin comment tout refaire de zéro.

Plan :
1. [Le projet en une page](#1-le-projet-en-une-page)
2. [Les outils utilisés](#2-les-outils-utilisés)
3. [Les notions indispensables](#3-les-notions-indispensables)
5. [Niveau 1 — Fondations : parler au simulateur](#5-niveau-1--fondations--parler-au-simulateur)
6. [Niveau 2 — Données : enregistrer et vérifier](#6-niveau-2--données--enregistrer-et-vérifier)
7. [Niveau 3 — L'intelligence : l'enseignant et le CNN](#7-niveau-3--lintelligence--lenseignant-et-le-cnn)
8. [Niveau 4 — Les scripts du pipeline](#8-niveau-4--les-scripts-du-pipeline)
9. [Niveau 5 — Les outils de diagnostic et d'optimisation](#9-niveau-5--les-outils-de-diagnostic-et-doptimisation)
10. [Ce qui se passe pendant une conduite, image par image](#10-ce-qui-se-passe-pendant-une-conduite-image-par-image)
11. [Données et modèles produits](#11-données-et-modèles-produits)
---

## 1. Le projet en une page

**Objectif** : faire faire des tours de piste à une voiture dans le simulateur DonkeyCar,
le plus vite possible sans sortir, avec un réseau de neurones convolutif (CNN) qui ne voit
**que la caméra** (+ la vitesse de la voiture).

**Idée centrale : l'apprentissage par imitation d'un « enseignant »**

```
                    ┌─────────────────────────── informations "privilégiées" ───┐
                    │  tracé de la piste, écart au centre (cte), cap (yaw)      │
                    ▼                                                           │
 SIMULATEUR ──► ENSEIGNANT (pilote automatique, donkey/autopilot.py) ──► commande correcte
     │                                                                          │ (étiquette)
     │ image caméra + vitesse                                                   │
     ▼                                                                          ▼
 ENREGISTREMENT (donkey/recorder.py)  ──►  data/raw/<session>/  (image, commande, télémétrie)
                                                     │
                                                     ▼
                                ENTRAÎNEMENT (scripts/train_cnn.py, PyTorch, CPU)
                                                     │
                                                     ▼
                                        models/cnn_fast_v1.pt
                                                     │
 SIMULATEUR ──► image + vitesse ──► CNN (donkey/cnn.py) ──► direction, accélérateur, frein
```

- L'**enseignant** est un programme qui « triche » : il connaît le tracé exact de la piste
  et la position de la voiture par rapport au centre. Il conduit très bien.
- On l'enregistre en train de conduire : pour chaque image de la caméra, on garde la
  commande qu'il a donnée.
- Le **CNN** apprend à prédire cette commande **à partir de l'image seule**. Une fois
  entraîné, il conduit sans jamais voir le tracé ni le cte.

Pourquoi pas de conduite humaine ? Tu n'es pas un pilote expert dans le simulateur ; un
CNN qui imite un humain ne peut pas faire mieux que lui. L'enseignant, lui, est régulier
et rapide, et on peut l'optimiser automatiquement.

Pourquoi pas d'apprentissage par renforcement (RL) ? Le simulateur tourne en temps réel
(20 images/s) sur un portable 2 cœurs sans GPU : un RL sur images demande des centaines
de milliers à des millions d'étapes, soit des jours. Analyse détaillée faite au début du
projet.

**Résultat** : `cnn_fast_v1` fait **37 tours sur 37 en 10 min, 0 sortie, 15,76 s par tour**
(l'enseignant optimisé fait 15,43 s).

---

## 2. Les outils utilisés

| Outil | Rôle dans le projet | Pourquoi lui |
|---|---|---|
| **DonkeySim** (`donkey_sim.exe`, Unity) | Le simulateur : piste, voiture, physique, caméra | Simulateur officiel du projet DonkeyCar |
| **gym_donkeycar 1.3.1** | Fournit `SDClient`, la classe de base qui gère la connexion TCP au simulateur | On réutilise seulement la couche réseau ; tout le reste est à nous |
| **Python 3.10**, env. conda `donkeycar_env` | Langage | Compatible avec gym_donkeycar |
| **NumPy** | Tableaux d'images et de nombres | Standard |
| **Pillow (PIL)** | Décoder les images PNG/JPEG, redimensionner | Simple, pas besoin d'OpenCV |
| **pygame** | Fenêtre de conduite manuelle au clavier (optionnel) | Clavier + affichage d'images faciles |
| **PyTorch 2.14 (CPU)** | Définir, entraîner et exécuter le CNN | Standard, fonctionne sans GPU |
| **git** | Historique du projet | — |

### Comment Python parle au simulateur

Le simulateur ouvre un serveur **TCP sur `127.0.0.1:9091`**. On échange des **messages JSON**,
un objet par message, avec un champ `msg_type`.

Messages qu'on **envoie** (tous vérifiés pendant le projet) :

| `msg_type` | Effet |
|---|---|
| `get_scene_names` / `load_scene` / `exit_scene` | Lister / charger / quitter une piste |
| `racer_info`, `car_config` | Nom du pilote, carrosserie et couleur de la voiture |
| `cam_config`, `cam_config_b`, `lidar_config` | Réglages caméra (taille, encodage, champ de vision), caméra 2 et lidar (désactivés) |
| `control` | **La commande** : `steering` (−1..1), `throttle` (−1..1), `brake` (0..1) |
| `reset_car` | Remet la voiture sur la ligne de départ |
| `node_position` (+ `index`) | Demande la position et l'orientation d'un point du tracé |

Messages qu'on **reçoit** :

| `msg_type` | Contenu |
|---|---|
| `telemetry` | 20 fois par seconde : image (base64), vitesse, cte, cap, position, collision, etc. |
| `car_loaded`, `scene_selection_ready`, `scene_names` | États du simulateur |
| `collision_with_starting_line` | Passage sur la ligne de départ (aussi envoyé après un reset !) |
| `node_position` | Réponse : `pos_x`, `pos_z` et quaternion `Qx..Qw` d'un point du tracé |

---

## 3. Les notions indispensables

### La télémétrie (ce que le simulateur nous dit)

| Champ | Signification | Utilisé par |
|---|---|---|
| `image` | Image caméra 320×240 PNG | CNN |
| `speed` | Vitesse de la voiture | CNN (entrée), enseignant |
| `cte` | **Cross-Track Error** : distance latérale au centre de la piste. Signe : négatif d'un côté, positif de l'autre. \|cte\| > 3 ≈ bord de piste | Enseignant, mesures |
| `yaw` | Cap de la voiture en degrés (0–360) | Enseignant |
| `activeNode` / `totalNodes` | Index du point du tracé le plus proche (0..146) / nombre de points (147) | Enseignant, comptage des tours |
| `hit` | Objet touché (`none` sinon) | Détection de collision |
| `time` | Horloge du simulateur, pas de 50 ms → **l'horodatage de référence** | Partout |
| `steering_angle`, `throttle` | « Écho » : la commande que le simulateur applique **en ce moment** (≈ 2 images après l'envoi) | Vérifications |
| `pos_*`, `vel_*`, `gyro_*`, `accel_*`, `roll`, `pitch`, odométrie | Physique détaillée | Enregistrées, peu utilisées |

### Le tracé de la piste (les « nœuds »)

La piste est décrite par **147 points** (nœuds) espacés d'environ 2,66 unités, le long de
la ligne centrale. Le message `node_position` donne leur position. On en déduit la
**direction de la piste** à chaque endroit et **les virages à venir**.

Attention : les positions des nœuds ne sont **pas** dans le même repère que `pos_x/pos_z`
de la télémétrie (vérifié). Par contre l'orientation des nœuds et `yaw` sont dans le même
repère (vérifié : voiture à 270,0° au départ, nœud 0 à 269,5°). C'est pour cela que
l'enseignant n'utilise que des **angles**, jamais de comparaisons de positions.

### Apprentissage par imitation (« behavioral cloning »)

On apprend une fonction `image → commande` par **régression supervisée** : pour chaque
image enregistrée, la « bonne réponse » (étiquette) est la commande de l'enseignant.

Problème classique : **le décalage de distribution**. Le CNN fait de petites erreurs, se
retrouve dans des situations que l'enseignant n'a jamais montrées (un peu décalé, un peu
trop vite…), et ne sait pas quoi faire. Deux remèdes utilisés :

1. **Injection de bruit pendant la collecte** (idée connue sous le nom de *DART*) : on
   perturbe la direction **appliquée** à la voiture, mais on **enregistre la commande
   correcte** de l'enseignant. La voiture s'écarte, l'enseignant la ramène → le jeu de
   données contient des **récupérations**.
2. **DAgger** (*Dataset Aggregation*) : c'est le **CNN qui conduit**, l'enseignant regarde
   et enregistre ce qu'il aurait fallu faire dans chaque situation. On ajoute ces données
   et on réentraîne. Le CNN apprend à corriger ses propres erreurs.

### Le CNN

Un réseau de neurones convolutif : des **couches de convolution** extraient des motifs
de l'image (bords des lignes blanches, courbure de la route…), puis des **couches
denses** combinent ces motifs avec la vitesse pour produire 3 nombres : direction,
accélérateur, frein.

### Le temps réel

Le simulateur envoie 20 images/s, **quoi qu'on fasse**. Il ne nous attend pas. Si notre
programme est lent, des images s'accumulent : pour conduire il faut toujours utiliser
**la plus récente** ; pour enregistrer il faut les garder **toutes, dans l'ordre**.
Une commande met ≈ 2 images (100 ms) à être appliquée (mesuré).


---

## 5. Niveau 1 — Fondations : parler au simulateur

### 5.1 [donkey/config.py](../donkey/config.py) — tous les réglages au même endroit

**Rôle** : centraliser les paramètres. Chaque script importe ses valeurs par défaut d'ici,
et chaque session enregistre ces réglages dans son `meta.json` (traçabilité).

Contenu :
- **Chemins** : `PROJECT_ROOT`, `DATA_RAW_DIR` (`data/raw`), `DATA_PROCESSED_DIR` (`data/processed`).
- **Plages des commandes** : direction et accélérateur dans [−1, 1], frein dans [0, 1].
- **`SimConfig`** : adresse `127.0.0.1:9091`, piste `roboracingleague_1`, délais d'attente.
- **`CarConfig`** : voiture `car01` magenta, nom « Choupette 53 », pilote « Reda ».
  Les méthodes `racer_msg()` et `car_msg()` fabriquent les messages JSON correspondants.
- **`CameraConfig`** : **320×240, PNG, champ de vision par défaut (90°)**. `to_msg()`
  fabrique le message `cam_config`.
- `config_dict()` : transforme les 3 configs en dictionnaire pour `meta.json`.

Les classes sont des `@dataclass(frozen=True)` : des structures de données immuables
(on ne peut pas les modifier par erreur pendant l'exécution).

**Décision clé — la caméra 320×240** (mesurée avec `tools/probe_camera.py`) :
160×120 était trop flou pour voir la piste au loin ; 640×480 ralentissait le simulateur à
0,7× le temps réel ; 320×240 arrive à 20 images/s sans perte. Le CNN réduit ensuite
l'image à 64×160, mais **la caméra doit rester identique** entre collecte, entraînement
et conduite.

### 5.2 [donkey/telemetry.py](../donkey/telemetry.py) — transformer un message en objet

**Rôle** : convertir un paquet JSON de télémétrie (dictionnaire de chaînes et nombres)
en un objet Python propre et typé : `Telemetry`.

Points importants :
- **`Telemetry`** (dataclass) : un attribut par champ (`speed`, `cte`, `yaw`,
  `active_node`, `sim_time`…), plus :
  - `frame_id` : un compteur qu'**on** attribue (0, 1, 2…) pour détecter des images manquantes ;
  - `t_local`, `wall_time` : heure de réception locale (mesures de latence) ;
  - `image_bytes` : l'image **encodée**, telle que reçue (on ne la décode que si besoin).
- **`from_packet()`** : décode le base64 de l'image, lit la taille dans l'en-tête PNG
  (sans décoder les pixels : rapide), convertit chaque champ. Un champ absent devient
  `NaN` (nombres) ou `-1` (entiers) : une valeur manquante est visible, jamais inventée.
  Les champs inconnus vont dans `extra` au lieu d'être perdus.
- `roll` et `pitch` arrivent dans [0, 360) (359,99 veut dire −0,01) : `wrap_180()` les
  ramène dans [−180, 180).
- **`image()`** : décode l'image en tableau NumPy RGB `(240, 320, 3)` de `uint8`.
- **`to_record()`** : dictionnaire plat de tous les champs (pour une ligne CSV).
- `image_format()` : reconnaît PNG/JPEG grâce aux premiers octets du fichier.

### 5.3 [donkey/sim_client.py](../donkey/sim_client.py) — le client du simulateur ★★★

**Rôle** : la classe `DonkeyClient` gère toute la communication : connexion,
chargement de la piste, configuration, réception des images, envoi des commandes.
Elle hérite de `SDClient` (gym_donkeycar), qui fournit la socket TCP et un **thread**
qui lit les messages en continu.

**Deux threads travaillent en parallèle** :
- le **thread réseau** (créé par SDClient) reçoit chaque message et appelle `on_msg_recv()` ;
- le **thread principal** (ton script) lit les images et envoie les commandes.

Ils partagent des données, protégées par un verrou (`threading.Condition`).

#### Réception : `on_msg_recv(j)`
- `telemetry` → crée un `Telemetry`, le met dans `_latest` (la plus récente) **et** dans
  `_buffer` (une file FIFO de 200 images max), puis réveille le thread principal.
- `car_loaded`, `scene_selection_ready`, `scene_names` → des `threading.Event`
  qui débloquent les attentes de `start()`.
- `collision_with_starting_line` → ajouté à `lap_events`.
- `node_position` → stocké dans `track_nodes[index]`.
- **Ne lève jamais d'exception** (tout est dans un `try/except`) : sinon SDClient coupe
  la connexion.

#### Lire les images : deux méthodes, deux usages
- **`get_frame()`** : la plus ancienne image non lue (FIFO). **Aucune image n'est
  sautée** → indispensable pour **enregistrer** (une image ↔ une commande).
- **`wait_for_latest()`** : la plus récente, les anciennes sont jetées → indispensable
  pour **conduire** : on réagit toujours à la situation actuelle, même si le programme a
  pris du retard.
- `latest()` : regarder la dernière sans rien consommer ; `flush()` : vider la file ;
  `backlog` : nombre d'images en attente.

#### Envoyer : `send_json()` et `send_controls()`
- On utilise toujours `send_now` (envoi immédiat). La méthode `send()` de SDClient
  n'a qu'**une place** : un deuxième message envoyé trop vite écrase le premier (bug de
  l'ancien `test_client.py`).
- `send_controls(steering, throttle, brake)` **borne** les valeurs dans leurs plages et
  renvoie ce qui a été réellement envoyé (avec l'heure d'envoi `t_local`).

#### Démarrer : `start(force_reload=False)`
1. Attend un signe de vie : si une piste tourne déjà, le simulateur envoie `car_loaded` ;
   s'il est au menu, `scene_selection_ready`.
2. Si `force_reload` : `exit_scene`, puis attend que la télémétrie s'arrête (le
   simulateur n'envoie **pas** `scene_selection_ready` après `exit_scene` — observé).
3. Sinon, si rien n'est chargé : `_load_scene()` → `get_scene_names` (3 essais) puis
   `load_scene` et attend `car_loaded`.
4. **`configure()`** : envoie pilote, voiture, caméra 2 désactivée, lidar désactivé, puis
   caméra, avec de petites pauses. **Après** que la voiture existe (sinon la caméra est
   ignorée — c'était le bug de `test_client.py`).
5. **`verify_camera()`** : attend une image de la bonne taille (320×240), sinon erreur.
6. `flush()` : jette les images reçues pendant la configuration.

#### Le tracé : `fetch_track(total_nodes)`
Envoie `node_position` pour les 147 nœuds et attend toutes les réponses. Renvoie la
liste des nœuds (positions + orientation). Utilisé par l'enseignant.

#### Fermer : `close()` / `with DonkeyClient() as client:`
À la sortie, la voiture reçoit accélérateur 0 + frein 1, puis la connexion est fermée.
Le `with` garantit cet arrêt même en cas d'erreur.

---

## 6. Niveau 2 — Données : enregistrer et vérifier

### 6.1 [donkey/recorder.py](../donkey/recorder.py) — `SessionRecorder`

**Rôle** : enregistrer une **session** = un dossier qui ne sera jamais écrasé :

```
data/raw/2026-11-14_05-35-03_roboracingleague_1_fast_noise/
    images/000000.png, 000001.png, ...   l'image exacte reçue du simulateur
    records.csv                          une ligne par image (42 colonnes)
    meta.json                            réglages, versions, compteurs, épisodes
```

**L'étiquette** (la définition la plus importante) : la ligne *i* = l'image *i* **+ la
commande envoyée juste après avoir reçu cette image** (`cmd_steering`, `cmd_throttle`,
`cmd_brake`). C'est ce que le CNN apprendra à prédire.

Fonctionnement :
- **`record(tel, command)`** est appelé 20 fois/s par la boucle de conduite. Il ne touche
  **pas** au disque (trop lent) : il construit la ligne et la met dans une file en mémoire
  `_pending`.
- Les échantillons plus vieux que `hold_seconds` partent vers un **thread d'écriture**
  (`_writer_loop`) qui écrit **l'image puis la ligne CSV** — ainsi une ligne ne pointe
  jamais vers une image absente, même après un plantage.
- **`discard_last(seconds)`** : supprime les dernières secondes encore en mémoire
  (« annuler » une erreur de conduite en manuel).
- **`new_episode(reason, sim_time)`** : à chaque reset, le numéro d'épisode augmente ;
  le temps du reset est noté pour vérifier le « cooldown » (voir §13).
- **`close()`** : écrit le reste, arrête le thread, finalise `meta.json` (statut
  `complete`, nombre d'images, épisodes…).
- Le nom du dossier est rendu unique par un `mkdir` atomique (`_2`, `_3` si collision).
- `meta.json` est écrit de façon atomique (fichier `.tmp` puis renommage).
- `COLUMN_DOC` documente chaque colonne ; il est copié dans `meta.json`.

Colonnes de `records.csv` (principales) : `sample_id`, `image`, `episode_id`, `mode`
(`normal` / `recovery` pendant le bruit), `lap_count`, `cmd_steering`, `cmd_throttle`,
`cmd_brake`, `cmd_latency_ms`, `frame_id`, `sim_time`, `echo_steering`, `speed`, `cte`,
`hit`, `active_node`, positions, vitesses, angles…

### 6.2 [donkey/session.py](../donkey/session.py) — relire et valider

- **`load_session(path)`** → `(meta, rows)` : lit `meta.json` et le CSV en convertissant
  chaque colonne dans le bon type (sans pandas).
- **`validate_session(path)`** : une série de contrôles, chacun PASS/FAIL :
  statut `complete`, nombre de lignes = compteur de `meta.json`, en-tête CSV correct,
  `sample_id` contigus, chaque ligne a son image, pas d'image orpheline, toutes les
  images se décodent, une seule taille d'image = celle de la config caméra, `sim_time` et
  `frame_id` strictement croissants, commandes dans leurs plages, épisodes croissants,
  **aucune image enregistrée pendant le cooldown après un reset**. Infos (sans échec) :
  trous temporels, vitesse du simulateur par rapport au temps réel.
- `print_report()` : affichage lisible.

---

## 7. Niveau 3 — L'intelligence : l'enseignant et le CNN

### 7.1 [donkey/autopilot.py](../donkey/autopilot.py) — l'enseignant ★★★

#### `AutopilotSettings` — les réglages
Valeurs de l'enseignant rapide (`models/profiles/teacher_fast_v1.json`, trouvées par
`tools/tune_teacher.py`) :

| Réglage | Valeur | Rôle |
|---|---|---|
| `lookahead` | 3 | Viser la direction de la piste 3 nœuds plus loin |
| `lookahead_per_speed` | 0,4 | + 0,4 nœud de visée par unité de vitesse |
| `k_heading` | 0,0383 | Direction par degré d'écart de cap (≈ 1/26) |
| `k_cte` | 0,2 | Correction par unité d'écart au centre |
| `cte_sign` | −1 | Convention de signe du cte (mesurée) |
| `speed_lookahead` | 7 | Nombre de nœuds regardés pour anticiper un virage |
| `v_max` / `v_min` | 5,5 / 2,75 | Vitesse cible en ligne droite / dans le virage le plus serré |
| `turn_full` | 80° | Virage considéré « le plus serré » |
| `k_speed` | 0,5 | Accélérateur par unité d'écart de vitesse |
| `throttle_ff` | 0,25 | Accélérateur de base |
| `max_throttle` | 1,0 | Plafond d'accélérateur |
| `brake_margin` | 0,4 | Freiner si plus rapide que la cible + 0,4 |
| `brake_gain` | 1,0 | Frein = 1,0 × excès de vitesse (max 1) |

#### `TrackAutopilot` — la logique

À la création, il reçoit les 147 nœuds et calcule la **direction de chaque segment**
`seg[i]` (angle de i vers i+1, `atan2(dx, dz)` en degrés — même convention que `yaw`).

À chaque image, `__call__(tel)` calcule :

**1. La direction (steering)**
```
look        = lookahead + round(lookahead_per_speed × vitesse)     # ex. 3 + round(0,4×3) = 4
cible       = direction de la corde nœud_actuel → nœud_actuel + look
erreur_cap  = cible − yaw          (ramené dans [−180, 180])
direction   = k_heading × erreur_cap  +  cte_sign × k_cte × cte     (bornée à [−1, 1])
```
- Viser **plusieurs nœuds en avant** fait tourner **avant** le virage (anticipation) ;
  plus on va vite, plus on vise loin.
- Le terme cte ramène doucement vers le centre.
- Exemple : cap 10° à gauche de la cible → 0,0383 × 10 = 0,38 de direction à droite.

Pourquoi pas un simple correcteur sur le cte ? Essayé en premier : il ne réagit qu'une fois
la voiture déjà décalée, trop tard dans les virages → sorties systématiques.

**2. La vitesse cible**
```
virage   = plus grand changement de direction sur les 7 prochains segments (degrés)
v_cible  = v_max − (v_max − v_min) × min(1, virage / turn_full)
```
Ligne droite → 5,5 ; virage de 80° ou plus en vue → 2,75 ; entre les deux, proportionnel.
(Option `profile` : une vitesse cible par nœud, apprise par `tools/optimize_profile.py`.)

**3. Accélérateur et frein**
```
accélérateur = throttle_ff + k_speed × (v_cible − vitesse)      (borné à [0, max_throttle])
excès        = vitesse − (v_cible + brake_margin)
frein        = min(1, brake_gain × excès) si excès > 0, et alors accélérateur = 0
```
C'est un **régulateur proportionnel** de vitesse avec freinage progressif.
(Ancien comportement, `brake_gain = 0` : frein fixe 0,3 — utilisé par `cnn.pt`.)

#### `SteeringNoise` — le bruit pour apprendre les récupérations
Toutes les ~4 s (aléatoire), une impulsion de direction de ±0,2 à ±0,4 pendant ~0,5 s,
plus un petit bruit permanent (écart-type 0,05). `active` indique si une impulsion est en
cours (la colonne `mode` vaut alors `recovery`).

### 7.2 [donkey/cnn.py](../donkey/cnn.py) — le CNN ★★★

#### Le prétraitement `preprocess(img)` — identique à l'entraînement et en conduite
```
image 240×320  ──recadrage lignes 80 à 215──►  135×320  ──redimension bilinéaire──►  64×160
```
- On enlève le haut (plafond, murs : inutile) et le bas (capot magenta de la voiture).
- On réduit à 64×160 : 7,5 fois moins de pixels → calcul rapide sur CPU, assez de détail
  pour voir les lignes.
- **Règle d'or** : la *même* fonction sert à l'entraînement (`train_cnn.py`) et en
  conduite (`CnnDriver`). Si les deux différaient, le CNN verrait en conduite des images
  différentes de celles apprises.

#### L'architecture `DriveNet` (inspirée de PilotNet de NVIDIA)

```
Entrée image : 3 × 64 × 160 (RGB, valeurs 0..1)
 bloc 1 : conv 5×5, pas 2, 24 filtres  → 24 × 32 × 80
 bloc 2 : conv 5×5, pas 2, 32 filtres  → 32 × 16 × 40
 bloc 3 : conv 3×3, pas 2, 48 filtres  → 48 ×  8 × 20
 bloc 4 : conv 3×3, pas 1, 64 filtres  → 64 ×  8 × 20
 bloc 5 : conv 3×3, pas 1, 64 filtres  → 64 ×  8 × 20
 aplatir                               → 10 240 nombres
 + vitesse (divisée par 5)             → 10 241
 dense 100 + ReLU + Dropout 0,3
 dense 50 + ReLU
 dense 3  → [direction, accélérateur, frein]
```
- Chaque bloc = **Convolution + BatchNorm + ReLU**. La convolution détecte des motifs
  locaux ; BatchNorm stabilise l'apprentissage ; ReLU ajoute la non-linéarité. Le
  **pas 2** divise la taille par 2 (on résume l'image).
- La **vitesse** est ajoutée après les convolutions : une seule image ne dit pas à quelle
  vitesse on va, or il faut freiner différemment selon la vitesse (c'était le problème du
  virage après la ligne de départ).
- **Dropout** (0,3) : éteint aléatoirement 30 % des neurones pendant l'entraînement pour
  éviter le par-cœur.
- **Sorties** : `tanh` → direction dans [−1, 1] ; `sigmoid` → accélérateur dans [0, 1] ;
  la 3e sortie brute (logit) → frein, transformée par `sigmoid`.
- ≈ **1,1 million de paramètres** (dont ~1 million dans la première couche dense).
  Inférence ≈ 7 ms par image sur ton CPU.

#### `to_tensor()` 
Tableau NumPy `(N, H, W, 3)` d'entiers 0..255 → tenseur PyTorch `(N, 3, H, W)` de réels
0..1 (PyTorch veut les canaux en premier).

#### `CnnDriver` — conduire avec un modèle
- Charge le fichier `.pt` (poids + métadonnées), passe en mode évaluation (`eval()` :
  désactive Dropout, BatchNorm en mode figé), 2 threads CPU.
- `__call__(tel)` : `preprocess` → CNN → (direction, accélérateur, frein).
- **Deux modes de frein** selon les métadonnées du modèle :
  - `binary` (ancien `cnn.pt`) : si la probabilité de freiner > 0,5 → frein 0,3 ;
  - `regression` (modèles `fast`) : la sortie *est* l'intensité du frein (0..1),
    ignorée sous 0,05.
  - Si on freine, l'accélérateur est mis à 0.

### 7.3 [donkey/controllers.py](../donkey/controllers.py) — le clavier (optionnel)

`KeyboardController` transforme l'état des touches en commande **progressive** : la
direction monte vers ±1 à 2 unités/s (une touche tapée = petite correction, maintenue
0,5 s = braquage complet) et revient au centre quand on relâche. Régulateur de vitesse
(« cruise ») optionnel. Logique pure, sans pygame : testable seule. Utilisé uniquement
par `drive_manual.py`.

---

## 8. Niveau 4 — Les scripts du pipeline

### 8.1 [scripts/run_driver.py](../scripts/run_driver.py) — conduire, enregistrer, mesurer ★★★

Le script principal. Il sert à **trois** choses selon les options :

| Usage | Commande |
|---|---|
| Tester l'enseignant | `python scripts/run_driver.py --profile models/profiles/teacher_fast_v1.json --minutes 2` |
| Collecter des données | `... --profile ... --record --noise --minutes 10` |
| Conduire avec le CNN | `python scripts/run_driver.py --driver cnn --minutes 3` |
| DAgger | `python scripts/run_driver.py --driver cnn --record --tag dagger --profile ...` |

Déroulement de `main()` :
1. **Options** (`argparse`) et réglages de l'enseignant (`--set clé=valeur` modifie
   n'importe quel réglage ; `--profile` charge un fichier de réglages).
2. Si `--driver cnn` : charge le modèle (`CnnDriver`).
3. Connexion (`DonkeyClient.start()`), récupération du tracé (`fetch_track`), création de
   l'**enseignant** — **toujours**, même quand le CNN conduit : il sert d'étiquette en
   DAgger.
4. Si `--record` : crée un `SessionRecorder`.
5. `do_reset()` : reset de la voiture + début d'un cooldown d'1 s (temps simulateur).
6. **La boucle**, une itération par image (voir §10 pour le détail).
7. **Bilan** JSON : tours, temps, tours propres, sorties, collisions, vitesse et cte
   moyens/max, images sautées. `--json-out` l'ajoute à un journal.
8. Si enregistrement : fermeture et validation automatique de la session.

`LapCounter` — compter les tours sans se tromper : un tour est compté quand `active_node`
passe de la fin de la piste (≥ 141) au début (≤ 5) **après avoir visité au moins 80 % des
nœuds**. Le message `collision_with_starting_line` n'est **pas** utilisé, car le simulateur
l'envoie aussi quand on remet la voiture sur la ligne (faux tours).

Détections :
- **collision** : `hit != "none"` → reset ;
- **sortie de piste** : `|cte| > 3` → reset ;
- **bloquée** : vitesse < 0,3 pendant 3 s → reset ;
- **tour propre** : tour sans aucune de ces erreurs.

### 8.2 [scripts/train_cnn.py](../scripts/train_cnn.py) — entraîner le CNN ★★★

```
python scripts/train_cnn.py data/raw/<session1> data/raw/<session2> --epochs 8 --out models/x.pt
python scripts/train_cnn.py <sessions> --init models/cnn_fast_v1.pt --epochs 4 --lr 5e-4 --out models/y.pt
```

Étapes :
1. **Sélection** des sessions (toutes celles de `data/raw` si aucune n'est donnée) et
   **validation** rapide (les sessions invalides sont ignorées).
2. **Chargement avec cache** (`load_one`) : chaque image est décodée et prétraitée
   (`preprocess`), puis tout est sauvé dans `data/processed/<session>_v2.npz`
   (`X` images 64×160, `V` vitesses, `Y` = direction, accélérateur, frein). La première
   fois ≈ 2–3 min par session, ensuite quelques secondes.
3. **Découpage entraînement / validation par le temps** : les **derniers 15 %** de chaque
   session servent à la validation. Pas un tirage aléatoire : deux images consécutives
   sont presque identiques, un tirage aléatoire mettrait des « jumelles » des deux côtés
   et la validation serait trompeusement bonne.
4. **`--init`** : repartir des poids d'un modèle existant (couches de même forme
   seulement) → apprentissage beaucoup plus rapide.
5. **Fonction de perte** (ce que l'entraînement minimise) :
   ```
   perte = MSE(direction) + 0,5 × MSE(accélérateur) + 0,1 × perte_frein
   ```
   - MSE = erreur quadratique moyenne (régression).
   - Frein, mode `regression` : MSE sur l'intensité, les images où l'on freine sont
     **pondérées davantage** (elles sont rares : ~5 % dans la première session), sinon le réseau apprendrait à ne
     jamais freiner. Mode `binary` : entropie croisée binaire pondérée.
6. **Augmentation** : luminosité et contraste aléatoires (×0,7–1,3, ±0,1) à chaque
   lot, pour résister aux variations d'éclairage.
7. **Optimiseur Adam**, taux d'apprentissage 1e-3 avec **décroissance cosinus**, lots de
   64 images, `--epochs` passages sur les données.
8. Après chaque époque : erreurs sur la validation ; **le meilleur modèle est sauvegardé**
   (`*saved`), avec ses métadonnées (sessions, époque, mode de frein, erreurs). Historique
   dans `<modèle>_train.json`.

Lire les chiffres affichés :
- `steer RMSE 0.12` : erreur typique de direction de 0,12 sur une échelle de −1 à 1.
  Référence : un modèle qui répond toujours la moyenne aurait l'écart-type des données
  (≈ 0,65, affiché à la fin).
- `brake acc 95 %` : le modèle sait s'il faut freiner ou non dans 95 % des images.
- **Attention** : une bonne erreur de validation ne garantit pas une bonne conduite. Le
  seul vrai test est la conduite (`run_driver.py`).

### 8.3 [scripts/validate_session.py](../scripts/validate_session.py)

`python scripts/validate_session.py --latest | --all | <dossier>` : rapport de
`validate_session()` + statistiques : réglages caméra, modes, épisodes, accélérateur et
vitesse moyens, collisions, **histogramme de la direction**. Avec `--all`, avertit si des
sessions de caméras différentes seraient mélangées.

### 8.4 [scripts/drive_manual.py](../scripts/drive_manual.py) (optionnel)

Fenêtre pygame : image caméra agrandie + panneau d'infos. Conduite au clavier
(`KeyboardController`), R = enregistrer, T = marquer « récupération », Retour arrière =
annuler les 5 dernières secondes, Espace = reset, Échap = quitter (la session est validée).
Développé au début, puis abandonné comme source principale de données (conduite humaine
pas assez bonne).

---

## 9. Niveau 5 — Les outils de diagnostic et d'optimisation

| Outil | Ce qu'il fait | Ce qu'on en a appris |
|---|---|---|
| [tools/probe_telemetry.py](../tools/probe_telemetry.py) | Se connecte, enregistre tous les messages, teste caméra, direction, accélérateur, frein, reset | Liste exacte des champs, 20 Hz, écho de commande décalé de ~2 images, `cam_config` à envoyer après le chargement |
| [tools/check_sim_client.py](../tools/check_sim_client.py) | Test automatique de `DonkeyClient` (400 images) | Aucune image perdue, taille correcte, écho exact |
| [tools/check_recorder.py](../tools/check_recorder.py) | Test automatique du `SessionRecorder` | Écriture fiable, cooldown nécessaire après reset |
| [tools/probe_camera.py](../tools/probe_camera.py) | Compare plusieurs réglages caméra (taille, champ de vision, position) | Choix du 320×240 |
| [tools/probe_dynamics.py](../tools/probe_dynamics.py) | Accélération et freinage sur la ligne droite | Accélérateur 1,0 → 3,75 en 2 s ; frein 1,0 trois fois plus efficace que 0,3 |
| [tools/optimize_profile.py](../tools/optimize_profile.py) | Apprend une vitesse cible par nœud, tour après tour (baisse avant les points risqués, hausse ailleurs) | Échec instructif : plus de vitesse cible ≠ tour plus rapide |
| [tools/tune_teacher.py](../tools/tune_teacher.py) ★★ | **Recherche coordonnée** des réglages de l'enseignant | 16,65 → 15,43 s |
| [tools/eval_teacher_candidates.py](../tools/eval_teacher_candidates.py) | Évalue une liste de variantes de réglages | Plateau ≈ 15,4 s |

### Comment marche `tune_teacher.py` (recherche coordonnée)
```
meilleur = réglages de départ ; score = évaluer(meilleur)
répéter :
    pour chaque paramètre p :
        essayer p + pas, puis p − pas
        évaluer : reset, 1 tour lancé depuis l'arrêt + 2 tours lancés
        score = moyenne des 2 tours lancés
        rejeté si sortie de piste ou |cte| max > 2,2  (marge de sécurité)
        si score meilleur de plus de 0,05 s → on garde, et on continue sur ce paramètre
jusqu'à ne plus rien améliorer ou épuiser le budget d'essais
```
C'est de l'**optimisation boîte noire** : on ne sait pas calculer le temps au tour à partir
des réglages, on ne peut que l'essayer. Chaque essai coûte ~1 minute de simulation.

---

## 10. Ce qui se passe pendant une conduite, image par image

`python scripts/run_driver.py --driver cnn` — à chaque tour de boucle (≈ toutes les 50 ms) :

```
1. tel = client.wait_for_latest()        ← la plus récente image (CNN) ;
                                           get_frame() si l'enseignant conduit (pour tout enregistrer)
2. compter les images sautées (trous dans frame_id)
3. cooldown ? (moins d'1 s de temps simulateur depuis un reset, ou `hit` encore actif)
4. enseignant(tel) → commande « correcte » (étiquette)
5. CNN(tel)       → preprocess(image) → DriveNet(image, vitesse/5) → direction, accélérateur, frein
6. client.send_controls(direction, accélérateur, frein)
7. si cooldown : on s'arrête là pour cette image
8. si enregistrement : recorder.record(tel, étiquette de l'enseignant)
9. statistiques : vitesse, |cte|
10. LapCounter : tour terminé ? → afficher le temps, tour propre ?
11. collision / sortie / bloquée → reset (+ nouvel épisode)
```

Avec l'enseignant en collecte : à l'étape 5, c'est l'enseignant qui conduit, et
`SteeringNoise` ajoute une perturbation à la direction **envoyée** — l'étiquette (étape 8)
reste la commande **propre**.

---

## 11. Données et modèles produits

### Sessions (`data/raw/`)
| Session | Images | Enseignant | A servi à |
|---|---|---|---|
| `…_autopilot_noise` | 11 897 | lent (19,6 s) + bruit | `cnn.pt` |
| `…_dagger1` | 4 485 | DAgger sur le 1er CNN | `cnn.pt` |
| `…_dagger2` | 5 972 | DAgger sur `cnn.pt` | (essai v4, non retenu) |
| `…_fast_noise` | 11 679 | rapide (15,43 s) + bruit | `cnn_fast_v1` |
| `…_fast_dagger1` | 5 956 | DAgger sur `cnn_fast_v1` | `cnn_fast_v2` |
| `…_aggr_noise` | 9 207 | agressif (15,3 s) + bruit | `cnn_aggr_v1` |

### Modèles (`models/`)
| Modèle | Frein | Tour lancé | Fiabilité |
|---|---|---|---|
| `cnn.pt` | binaire 0,3 | 19,67 s | 30/30 en 10 min |
| **`cnn_fast_v1.pt`** | progressif | **15,76 s** | **37/37 en 10 min** |
| `cnn_fast_v2.pt` | progressif | 15,83 s | 37/37 en 10 min |
| `cnn_aggr_v1.pt` | progressif | 15,55 s (meilleur) | 3 sorties en 4 min |

Fichier `.pt` = dictionnaire `{"model": poids, "meta": {sessions, époque, brake_mode,
erreurs}}`. Les réglages d'enseignant sont dans `models/profiles/*.json`.

### Journaux (`docs/experiments/`)
Une ligne JSON par essai : `runs.jsonl` (toutes les conduites évaluées), `tune_*.jsonl`
(recherche de réglages), `profile_*.jsonl` (profil de vitesse), `candidates_*.jsonl`.


