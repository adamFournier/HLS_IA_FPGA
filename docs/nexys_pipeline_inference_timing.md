# Nexys A7 — latence réelle d'inférence FPGA (pipeline)

## Ce que mesure le RTL

Dans `src/rtl/src_nexys_pipeline/soc_mlp.sv`, un compteur matériel
`active_compute_cycles` avance **uniquement lorsque `phase_active=1`**.
Les pauses entre phases, en particulier les attentes UART, ne sont donc pas
comptées. Un timestamp d'entrée accompagne chaque image d'une couche à l'autre
avec un registre à décalage de longueur `NUM_LINEAR`.

- **Départ** : `start_phase_pulse` de la première couche pour cette image.
- **Arrivée** : `last_layer_done_pulse` de la dernière couche pour **la même image**.
- **Latence d'inférence** : différence de ces deux timestamps matériels,
  en cycles actifs. Les premiers `NUM_LINEAR-1` résultats correspondent au
  remplissage du pipeline ; Python les ignore.
- **Durée de phase** : compteur séparé `phase_cycles`, permettant de mesurer
  le temps matériel d'un lot sans double-comptage des calculs parallèles.

**Important :** il ne s'agit pas du temps chronologique continu entre
l'arrivée d'une image sur la liaison UART et le dernier octet transmis en sortie.
Les pauses entre phases sont explicitement exclues. Ce choix donne la durée
d'inférence **matérielle hors communications**, tandis que Python affiche aussi
le temps complet **avec UART**.

## Protocole UART (nouveau bitstream pipeline)

Entrée : `784 * int16` (1 568 octets), little-endian, inchangée.

Sortie du pipeline : `10 * int16 + 2 * uint32` = **28 octets** :

1. 20 octets : les dix scores signés (little-endian).
2. 4 octets : **latence d'inférence de cette image**, mesurée dans le FPGA.
3. 4 octets : **durée de calcul actif de la phase**, mesurée dans le FPGA.

Les deux timers utilisent des cycles à 100 MHz, soit `cycles / 100_000` ms.
Les sorties des premières phases ne sont pas des prédictions valides ;
le script les élimine au moyen de `NUM_LINEAR-1`.

Le RTL **séquentiel** ne change pas : 20 octets de scores + 4 octets de
compteur = 24 octets.

## Calcul des performances Python

Le script `notebooks/04_push_data_nexys.py` lit directement la latence
matérielle par image ; il ne la reconstruit plus en multipliant la durée
d'une phase par le nombre de couches.

Pour `N` images et `L` couches, il transmet `N+L-1` trames, les dernières
servant à vider le pipeline.

- **Latence par image** : le compteur mesuré dans les 4 octets dédiés.
- **Temps du lot sans UART** : somme des phases complètes **avant la dernière**
  + la partie réellement nécessaire de la dernière phase, jusqu'à la fin de
  la dernière image utile (pas le calcul de la trame de vidage).
- **Débit du lot** : `N / temps_lot_inference`.
- **Débit de régime établi** : fréquence FPGA / durée moyenne d'une phase.
- **Temps total avec UART** : `time.perf_counter()` autour des transferts du
  dataset, incluant les transferts UART et l'exécution côté PC et FPGA, mais
  excluant l'ouverture du port.

Les CSV incluent `inference_latency_cycles`, `inference_latency_ms`,
`batch_inference_ms`, `batch_throughput_images_per_s`,
`raw_output_phase_cycles` et `total_elapsed_with_uart_s`.

## Re-générer et tester

**Il faut créer un nouveau bitstream : l'ancien ne transmet que 24 octets,
alors que Python attend désormais 28 octets en mode pipeline.**

1. Récupérer la branche `feature/nexys-pipeline` avec `git pull`.
2. Dans `notebooks/03_build_FPGA_project.ipynb`, mettre `target='nexys_a7'`
   et `architecture='pipeline'`, puis exécuter la génération avec
   `clean_generated=True` afin de copier les nouveaux templates vers
   `src/rtl/generated/src_nexys_pipeline`.
3. Générer le projet/bitstream Vivado et programmer la Nexys A7.
4. Exécuter `py notebooks/04_push_data_nexys.py --fpga-mode pipeline --port COM5 --baud 2000000 --max-samples 100`
   (adapter le port COM).
5. Vérifier les calculs de protocole et de batch via
   `py -m unittest discover -s tests -p test_nexys_pipeline_timing.py`.

Le code Python est testé sur des valeurs artificielles. La synthèse Vivado
et la mesure sur une Nexys physique restent à valider sur la carte.
