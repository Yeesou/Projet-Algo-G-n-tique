from Groupes import Groupes
import time

DURATION = 10

cities = 100
indiv_length = cities * 5
tournament_size = indiv_length // 10
mutation_rate = 1

France = Groupes(indiv_length, tournament_size, mutation_rate=mutation_rate)

#France.generate_random_city(cities)
France.generate_circle_city(cities)
France.generate_individus()

fitness_checkpoint = []

start = time.time()
i = 0

while time.time() - start < DURATION:
    enfant1, enfant2 = France.croisement(method='hx extended', mutation='2-opt alt')
    if i % 100 == 0:
        print("Iteration:", i)
        fitness_checkpoint.append(max(indiv.fitness for indiv in France.individus))
    i += 1

meilleur = max(France.individus, key=lambda indiv: indiv.fitness)
print(f"\nMeilleur individu: {meilleur.chemin} (Fitness: {meilleur.fitness}, Distance: {meilleur.total_distance})")

France.plot_result(meilleur, title="Évolution du meilleur chemin")
France.plot_evolution(fitness_checkpoint)