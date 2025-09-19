from Groupes import Groupes

cities = 50
indiv_length = cities * 5
tournament_size = indiv_length // 10
mutation_rate = .9

France = Groupes(indiv_length, tournament_size, mutation_rate=mutation_rate)

#France.generate_random_city(cities)
France.generate_circle_city(cities)
France.generate_individus()

fitness_checkpoint = []

for i in range(5000):
    enfant1, enfant2 = France.croisement()
    if i % 100 == 0:
        print("Generation", i, "Best fitness:", max(indiv.fitness for indiv in France.individus))
        fitness_checkpoint.append(max(indiv.fitness for indiv in France.individus))

meilleur = max(France.individus, key=lambda indiv: indiv.fitness)
print(f"\nMeilleur individu: {meilleur.chemin} (Fitness: {meilleur.fitness}, Distance: {meilleur.total_distance})")

France.plot_result(meilleur, title="Évolution du meilleur chemin")
#France.plot_evolution(fitness_checkpoint)