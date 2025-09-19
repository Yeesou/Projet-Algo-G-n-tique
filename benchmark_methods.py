from Groupes import Groupes
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

DURATION = 5
CITIES = 100
INDIV_LENGTH = CITIES * 5
TOURNAMENT_SIZE = INDIV_LENGTH // 10
MUTATION_RATE = 0.0
SEED = 80

METHODS = [
    'pmx',
    'cx',
    'erx',
    'ox',
    'hx',
    'hx extended',
]


def build_group(seed: int) -> Groupes:
    # Seed pour reproductibilité de la population initiale
    random.seed(seed)
    np.random.seed(seed)

    g = Groupes(INDIV_LENGTH, TOURNAMENT_SIZE, mutation_rate=MUTATION_RATE)
    # Villes en cercle -> coordonnées déterministes, pas besoin de seed
    g.generate_circle_city(CITIES)
    g.generate_individus()
    return g


def run_benchmark(method: str, duration: float, seed: int) -> Dict[str, Any]:
    g = build_group(seed)

    start = time.time()
    iters = 0
    best = max(g.individus, key=lambda indiv: indiv.fitness)
    # Historique (temps écoulé en secondes, fitness)
    best_fitness_history: List[Tuple[float, float]] = []
    # Échantillonnage à pas de temps régulier (pour un axe X en secondes)
    SAMPLE_STEP = 0.1  # en secondes
    next_sample = start

    while time.time() - start < duration:
        enfant1, enfant2 = g.croisement(method=method)
        # Mettre à jour le meilleur efficacement
        if enfant1.fitness > best.fitness:
            best = enfant1
        if enfant2.fitness > best.fitness:
            best = enfant2

        now = time.time()
        if now >= next_sample:
            best_fitness_history.append((now - start, best.fitness))
            next_sample += SAMPLE_STEP
        iters += 1

    # Final best
    final_best = max(g.individus, key=lambda indiv: indiv.fitness)
    if final_best.fitness > best.fitness:
        best = final_best

    return {
        'method': method,
        'iterations': iters,
        'best_fitness': best.fitness,
        'best_distance': best.total_distance,
        'best_path': best.chemin,
        'history': best_fitness_history,
    }


def main():
    print(f"Benchmark sur {DURATION}s, cities={CITIES}, l={INDIV_LENGTH}, k={TOURNAMENT_SIZE}, mutation_rate={MUTATION_RATE}, seed={SEED}")
    results = []
    for method in METHODS:
        print(f"\n--- Méthode: {method} ---")
        res = run_benchmark(method, DURATION, SEED)
        print(f"Itérations: {res['iterations']}")
        print(f"Meilleure distance: {res['best_distance']:.6f}")
        print(f"Meilleure fitness:  {res['best_fitness']:.8f}")
        results.append(res)

    # Résumé final trié par meilleure distance
    results_sorted = sorted(results, key=lambda r: r['best_distance'])
    print("\n===== Résumé (trié par meilleure distance) =====")
    for r in results_sorted:
        print(f"{r['method']:<3} | iters={r['iterations']:<6} | dist={r['best_distance']:.6f} | fit={r['best_fitness']:.8f}")

    # Tracé de l'évolution de la fitness au fil du temps
    plt.figure(figsize=(10, 6))
    for r in results:
        if r['history']:
            xs, ys = zip(*r['history'])
            plt.plot(xs, ys, label=r['method'])
    plt.title('Évolution de la fitness au fil du temps (s)')
    plt.xlabel('Temps (secondes)')
    plt.ylabel('Meilleure fitness')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig('benchmark_fitness.png', dpi=150)
        print("Figure sauvegardée: benchmark_fitness.png")
    except Exception as e:
        print(f"Erreur sauvegarde figure: {e}")
    plt.show()


if __name__ == '__main__':
    main()
