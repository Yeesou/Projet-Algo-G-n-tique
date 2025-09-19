import time
from Groupes import Groupes
import matplotlib.pyplot as plt

def plot_evolution_comparative(times1, fitness1, times2, fitness2, label1="France", label2="Espagne", display=True):
    """Affiche l'évolution comparative de la fitness au cours du temps"""
    plt.figure(figsize=(10, 6))
    plt.plot(times1, fitness1, 'b-', linewidth=2, label=label1)
    plt.plot(times2, fitness2, 'r-', linewidth=2, label=label2)
    plt.title("Évolution comparative de la fitness (en fonction du temps)", fontsize=14, fontweight="bold")
    plt.xlabel("Temps (s)", fontsize=12)
    plt.ylabel("Meilleure fitness", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if display: plt.show()


DURATION = 15

cities = 250
indiv_length = cities * 5
tournament_size = indiv_length // 10
mutation_rate = 1

France = Groupes(indiv_length, tournament_size, mutation_rate=mutation_rate)

#France.generate_random_city(cities)
France.generate_circle_city(cities)
France.generate_individus()

Espagne = France.copy()

fitness_fr, times_fr = [], []
fitness_esp, times_esp = [], []

# --- Boucle France ---
timer_start = time.perf_counter()
i = 0
while True:
    France.croisement(method='pmx', mutation='2-opt')
    elapsed = time.perf_counter() - timer_start
    best_fit_fr = max(indiv.fitness for indiv in France.individus)
    fitness_fr.append(best_fit_fr)
    times_fr.append(elapsed)
    if i % 100 == 0:
        print("[ FRANCE  ] Gen", i, "Fitness:", best_fit_fr)
    if time.perf_counter() - timer_start >= DURATION:
        break
    i += 1

print("")

# --- Boucle Espagne ---
timer_start = time.perf_counter()
i = 0
while True:
    Espagne.croisement(method='hx', mutation='2-opt alt')
    elapsed = time.perf_counter() - timer_start
    best_fit_esp = max(indiv.fitness for indiv in Espagne.individus)
    fitness_esp.append(best_fit_esp)
    times_esp.append(elapsed)
    if i % 100 == 0:
        print("[ ESPAGNE ] Gen", i, "Fitness:", best_fit_esp)
    if time.perf_counter() - timer_start >= DURATION:
        break
    i += 1

# Résultats finaux
meilleur_fr = max(France.individus, key=lambda indiv: indiv.fitness)
meilleur_esp = max(Espagne.individus, key=lambda indiv: indiv.fitness)

print(f"\nMeilleur FR: Fitness {meilleur_fr.fitness}, Distance {meilleur_fr.total_distance}")
print(f"Meilleur ESP: Fitness {meilleur_esp.fitness}, Distance {meilleur_esp.total_distance}")

France.plot_result(meilleur_fr, title="Chemin FR", display=False)
Espagne.plot_result(meilleur_esp, title="Chemin ESP", display=False)

# --- Comparaison temps ---
plot_evolution_comparative(times_fr, fitness_fr, times_esp, fitness_esp, label1="France (2-opt)", label2="Espagne (2-opt alt)", display=False)

plt.show()
