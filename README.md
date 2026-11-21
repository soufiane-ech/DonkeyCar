# DonkeyCar — conduite autonome rapide par CNN (simulateur)

Un CNN conduit la voiture **uniquement à partir de la caméra** (+ sa vitesse) :
direction, accélérateur et frein. Il a appris en imitant un pilote automatique
(« enseignant ») qui, lui, utilise les informations internes du simulateur
(tracé de la piste, écart au centre). Aucune conduite humaine n'est nécessaire.

**Pour tout comprendre :
[docs/GUIDE_COMPLET.md](docs/GUIDE_COMPLET.md).**

## Résultat final

Modèle recommandé : **`models/cnn_fast_v1.pt`** (modèle par défaut de `run_driver.py`).

| Évaluation de 10 min, CNN seul au volant | CNN v3 (avant) | **cnn_fast_v1** | cnn_fast_v2 |
|---|---|---|---|
| Tours / tours sans sortie | 30 / 30 | **37 / 37** | 37 / 37 |
| Sorties de piste, collisions | 0, 0 | **0, 0** | 0, 0 |
| Meilleur tour | 19,55 s | **15,70 s** | 15,75 s |
| Tour lancé moyen | ≈ 19,67 s | **15,76 s** | 15,83 s |
| Écart au centre moyen / max | 0,19 / 2,05 | 0,68 / 2,40 | 0,68 / 2,43 |

**Gain : −20 % sur le temps au tour (19,67 → 15,76 s), toujours sans aucune sortie.**
Une « sortie » = |cte| > 3 (critère prudent ; la voiture est nettement hors piste à |cte| ≈ 4).
Le premier tour (~17,3 s) part de l'arrêt.

Tour propre le plus rapide obtenu par un CNN : **15,55 s** (`cnn_aggr_v1`), mais ce
modèle sort 3 fois en 4 min → non retenu.

## Modèle multi-pistes : `models/cnn_multi_v1.pt`

Un seul CNN pour **roboracingleague_1, generated_track et circuit_launch**, fait de la même
façon que `cnn_fast_v1` (qui reste inchangé, le meilleur sur roboracingleague_1).

| Évaluation de 4 min, CNN seul | Tours sans sortie | Sorties | Tour lancé moyen | Enseignant |
|---|---|---|---|---|
| roboracingleague_1 | 14 / 14 | 0 | 16,00 s | 15,43 s |
| generated_track | 18 / 18 | 0 | 8,86 s | 8,20 s |
| circuit_launch | 15 / 15 | 0 | 14,97 s | 14,43 s |

Recette :
1. Enseignant réglé **par piste** (`tools/tune_teacher.py --scene ...`) :
   `models/profiles/teacher_generated_track.json` (9,85 → 8,2 s),
   `teacher_circuit_launch.json` (18,25 → 14,43 s), `teacher_fast_v1.json` pour roboracingleague_1.
2. Collecte avec bruit : generated_track 8 407 images, circuit_launch 11 695 images,
   + les sessions `fast_noise` et `fast_dagger1` de roboracingleague_1.
3. Entraînement (`--flip --color --balance --init models/cnn_fast_v1.pt`, 10 époques) :
   miroir gauche/droite, couleurs/saturation/ombres/bruit, chaque piste pèse autant.
4. Règle de l'enseignant appliquée au CNN : **pas de frein sous 2,7** (`min_brake_speed` dans
   le modèle). Sans elle, le CNN freinait jusqu'à l'arrêt au virage après le départ de
   generated_track et restait bloqué.

Particularités : sur `circuit_launch` la voiture démarre à côté du tracé (|cte| ≈ 4, route
large) → les sorties ne sont comptées qu'après avoir rejoint le centre. Sur `generated_track`
le simulateur tourne à ≈ 0,4× le temps réel sur ce PC (les temps restent en temps simulé).

```
python scripts/run_driver.py --driver cnn --model models/cnn_multi_v1.pt --scene generated_track --laps 3
```

## Approches testées (4 h, CPU seul)

Chronologie des temps au tour lancé (pilote automatique = enseignant) :

| Étape | Enseignant | CNN |
|---|---|---|
| Départ (session précédente) | 19,6 s | 19,67 s |
| Accélérateur jusqu'à 1,0 (au lieu de 0,7) | 18,1 s | – |
| + freinage proportionnel (jusqu'à 1,0) + visée plus lointaine à haute vitesse | 16,6 s | – |
| + réglages optimisés par recherche coordonnée (40 essais) | **15,43 s** | – |
| CNN entraîné sur cet enseignant (+ frein progressif) | | **15,75 s** |
| + 1 tour de DAgger (`cnn_fast_v2`) | | 15,83 s (pas mieux) |

1. **Sonde de dynamique** (`tools/probe_dynamics.py`). Accélérateur 1,0 → vitesse 3,75
   en 2 s ; à 0,7 seulement ~3,0. Frein 1,0 : arrêt depuis 3,7 en ~1,2 s ; le frein 0,3
   de l'ancien enseignant était 3× trop faible. → L'enseignant n'exploitait pas la voiture.
2. **Profil de vitesse appris virage par virage** (`tools/optimize_profile.py`,
   apprentissage itératif : on baisse la vitesse avant les points où la voiture s'écarte,
   on l'augmente ailleurs). Meilleur tour 16,55 s dès le départ, puis **les tours sont
   devenus plus lents** (jusqu'à 18,2 s) quand les vitesses cibles augmentaient :
   au-delà d'une certaine vitesse la voiture élargit et freine plus. Conclusion : il faut
   optimiser le **temps au tour** directement, pas l'écart au centre.
3. **Recherche coordonnée sur les réglages de l'enseignant** (`tools/tune_teacher.py`) :
   score = temps moyen de 2 tours lancés, rejet si sortie ou |cte| > 2,2 (marge pour le
   CNN). 16,65 → **15,43 s** en 40 essais. Une 2e recherche (marge 2,4) et des combinaisons
   (`tools/eval_teacher_candidates.py`) confirment un **plateau ≈ 15,4 s**.
4. **CNN sur l'enseignant rapide** : la 3e sortie du CNN devient une **intensité de frein**
   (régression 0–1) au lieu d'un frein tout-ou-rien. Entraîné sur 11 679 images (10 min
   de collecte avec bruit), en repartant des couches de vision du CNN précédent : 8 époques
   ≈ 9 min sur CPU.
5. **Accélérateur du CNN ×1,05 / ×1,10** : 15,6–15,65 s, gain ≈ 0,1 s seulement — la vitesse
   du CNN est limitée par les vitesses de virage apprises, pas par l'accélérateur.
6. **Enseignant agressif** (visée plus forte, 15,3 s, |cte| max 2,53) : avec bruit il sort
   16 fois en 8 min ; le CNN qui l'imite (`cnn_aggr_v1`) fait 15,55 s mais sort 3 fois en
   4 min. **Imiter un pilote sans marge donne un CNN sans marge.**

## Analyse

- **Limite du circuit** : les réglages qui descendent sous ~15,2 s coupent les virages
  (|cte| 2,6–3,0). Avec visée à 4 nœuds l'enseignant a fait un tour en **14,6 s**, puis
  est sorti. La limite physique de ce circuit avec cette voiture est donc ≈ 14,6–15 s ;
  le CNN retenu est à ~5–8 % de cette limite, avec une marge de sécurité.
- **Écart CNN / enseignant** : ~0,3 s par tour et ~+0,3 d'écart max au centre. Le CNN
  ajoute un peu d'imprécision ; il faut donc un enseignant avec marge (|cte| max ≤ 2,2).
- **DAgger** : utile quand le CNN rencontre des états jamais vus (virage après la ligne
  de départ, session précédente) ; ici le CNN était déjà fiable, DAgger n'a rien gagné.
- **Temps réel** : la conduite utilise toujours la dernière image (`wait_for_latest`) ;
  inférence 7 ms/image. Ne rien lancer de lourd sur la machine pendant la conduite.
- **Ce qui n'a pas été fait** : apprentissage par renforcement (trop coûteux en temps de
  simulation sur ce PC, voir l'analyse initiale), plusieurs circuits, sens inverse.

## Modèles (tous conservés)

| Fichier | Description |
|---|---|
| `models/cnn_fast_v1.pt` | **recommandé** — enseignant rapide, frein progressif (15,76 s, 0 sortie) |
| `models/cnn_fast_v2.pt` | v1 + 1 tour de DAgger (15,83 s, 0 sortie) |
| `models/cnn_aggr_v1.pt` | enseignant agressif (15,55 s meilleur tour, mais 3 sorties / 4 min) |
| `models/cnn.pt` | modèle précédent v3, frein tout-ou-rien (19,67 s, 0 sortie) |
| `models/profiles/teacher_fast_v1.json` | réglages de l'enseignant rapide (15,43 s) |
| `models/profiles/teacher_*.json`, `profile_*.json` | résultats des recherches |

Chaque `*_train.json` contient l'historique d'entraînement du modèle correspondant.

## Données

`data/raw/` (une session = un dossier : images PNG 320×240, `records.csv`, `meta.json`)

| Session | Images | Usage |
|---|---|---|
| `…_autopilot_noise`, `…_dagger1`, `…_dagger2` | 11 897 / 4 485 / 5 972 | enseignant lent (CNN v3) |
| `…_fast_noise` | 11 679 | enseignant rapide → cnn_fast_v1 |
| `…_fast_dagger1` | 5 956 | DAgger → cnn_fast_v2 |
| `…_aggr_noise` | 9 207 | enseignant agressif → cnn_aggr_v1 |

`data/processed/` = cache des images prétraitées (régénérable, supprimable).
`docs/experiments/` = journaux de toutes les expériences (une ligne JSON par essai ;
`runs.jsonl` = toutes les évaluations de conduite).

## Structure

```
donkey/                 bibliothèque
  config.py             réglages (simulateur, voiture, caméra 320x240)
  sim_client.py         client du simulateur (connexion, commandes, images, tracé)
  telemetry.py          décodage de la télémétrie
  recorder.py           enregistrement des sessions
  session.py            lecture et validation des sessions
  autopilot.py          enseignant (visée sur le tracé, profil de vitesse, freinage)
  cnn.py                CNN : prétraitement, modèle, pilote CNN
  controllers.py        contrôle clavier (conduite manuelle)
scripts/                pipeline
  run_driver.py         conduite (enseignant ou CNN), enregistrement, mesures
  train_cnn.py          entraînement du CNN
  validate_session.py   vérification des sessions
  drive_manual.py       conduite manuelle au clavier (optionnel)
tools/                  diagnostics et optimisation
  probe_dynamics.py     accélération / freinage de la voiture
  tune_teacher.py       recherche des meilleurs réglages de l'enseignant
  eval_teacher_candidates.py  évaluation de variantes de réglages
  optimize_profile.py   profil de vitesse par apprentissage itératif
  probe_telemetry.py, probe_camera.py, check_sim_client.py, check_recorder.py
```

## Utilisation

Lancer le simulateur (`donkey_sim.exe`), puis :

```
conda activate donkeycar_env

# conduite autonome par le meilleur CNN + mesures
python scripts/run_driver.py --driver cnn --minutes 3
python scripts/run_driver.py --driver cnn --laps 1          # un seul tour

# enseignant rapide seul (référence)
python scripts/run_driver.py --profile models/profiles/teacher_fast_v1.json --minutes 2

# refaire le pipeline rapide
python scripts/run_driver.py --profile models/profiles/teacher_fast_v1.json --record --noise --minutes 10
python scripts/train_cnn.py data/raw/<session>_fast_noise --init models/cnn.pt --epochs 8 --out models/cnn_new.pt
python scripts/run_driver.py --driver cnn --model models/cnn_new.pt --minutes 10

# chercher de meilleurs réglages pour l'enseignant
python tools/tune_teacher.py --init models/profiles/teacher_fast_v1.json --max-evals 30
```

Options utiles de `run_driver.py` : `--laps N`, `--throttle-scale`, `--set clé=valeur`
(réglages de l'enseignant), `--json-out fichier` (ajoute le bilan au journal).
