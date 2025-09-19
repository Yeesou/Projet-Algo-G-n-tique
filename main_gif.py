from Groupes import Groupes

import matplotlib.pyplot as plt

cities = 50
indiv_length = cities * 5
tournament_size = indiv_length // 10
mutation_rate = 1

France = Groupes(indiv_length, tournament_size, mutation_rate)
#France.generate_circle_city(cities)
France.generate_random_city(cities)
France.generate_individus()

print("Démarrage de l'animation...")
anim = France.animate_evolution(generations=5000)

if anim:
    plt.show()

# Afficher le résultat final
meilleur = max(France.individus, key=lambda indiv: indiv.fitness)
print(f"\nRésultat final: Distance = {meilleur.total_distance:.3f}")
France.plot_result(meilleur)