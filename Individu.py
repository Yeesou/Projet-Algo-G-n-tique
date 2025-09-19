import random

class Individu:
    def __init__(self, group, chemin):
        self.group = group
        self.chemin = chemin
        self.fitness = 0
        self.total_distance = 0
        self.calc_total_distance()
        self.fitness = 1 / float(self.total_distance)
        
    def calc_total_distance(self):
        total = 0
        for i in range(len(self.chemin)):
            ville1 = self.chemin[i]
            ville2 = self.chemin[(i + 1) % len(self.chemin)]
            total += self.group.distance_between(ville1, ville2)
        self.total_distance = total
        
    def mutation_2opt(self):
        n = len(self.chemin)
        if n < 4:
            return  # Pas assez de villes pour 2-opt
            
        # Sélectionner deux arêtes aléatoirement
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        # S'assurer que i et j sont différents et dans l'ordre
        if i == j:
            return
        if i > j:
            i, j = j, i
            
        # S'assurer qu'il y a au moins une ville entre les deux arêtes
        if j - i < 2:
            return
            
        # Les villes concernées (a, b) et (c, d)
        a = self.chemin[i]
        b = self.chemin[(i + 1) % n]
        c = self.chemin[j]
        d = self.chemin[(j + 1) % n]
        
        # Calculer les distances actuelles et nouvelles
        dist_ab = self.group.distance_between(a, b)
        dist_cd = self.group.distance_between(c, d)
        dist_ac = self.group.distance_between(a, c)
        dist_bd = self.group.distance_between(b, d)
        
        # Vérifier si le changement améliore le coût
        if dist_ab + dist_cd > dist_ac + dist_bd:
            # Inverser le segment entre i+1 et j (inclus)
            self.chemin[i+1:j+1] = reversed(self.chemin[i+1:j+1])
            self.calc_total_distance()
            self.fitness = 1 / float(self.total_distance)
    
    def mutation_3opt(self):
        n = len(self.chemin)
        if n < 6:
            return  # Pas assez de villes pour 3-opt
            
        # Sélectionner trois arêtes aléatoirement
        indices = random.sample(range(n), 3)
        indices.sort()
        i, j, k = indices
        
        # S'assurer qu'il y a au moins une ville entre chaque paire d'arêtes
        if j - i < 2 or k - j < 2 or (i + n - k) < 2:
            return
            
        # Les villes concernées (a, b), (c, d), (e, f)
        a = self.chemin[i]
        b = self.chemin[(i + 1) % n]
        c = self.chemin[j]
        d = self.chemin[(j + 1) % n]
        e = self.chemin[k]
        f = self.chemin[(k + 1) % n]
        
        # Calculer les distances actuelles et nouvelles pour les différentes reconnections
        dist_ab = self.group.distance_between(a, b)
        dist_cd = self.group.distance_between(c, d)
        dist_ef = self.group.distance_between(e, f)
        
        dist_ac = self.group.distance_between(a, c)
        dist_bd = self.group.distance_between(b, d)
        
        dist_ae = self.group.distance_between(a, e)
        dist_bf = self.group.distance_between(b, f)
        
        dist_ce = self.group.distance_between(c, e)
        dist_df = self.group.distance_between(d, f)
        
        # Liste des reconnections possibles
        reconnections = [
            (dist_ac + dist_bd + dist_ef, [(i + 1, j)]),  # Reconnect (a,c) and (b,d)
            (dist_ab + dist_ce + dist_df, [(j + 1, k)]),  # Reconnect (c,e) and (d,f)
            (dist_ae + dist_bf + dist_cd, [(i + 1, k)]),  # Reconnect (a,e) and (b,f)
            (dist_ac + dist_ce + dist_bd + dist_df - dist_ab - dist_cd - dist_ef, [(i + 1, j), (j + 1, k)]),  # Double reconnection
            (dist_ae + dist_ac + dist_bd + dist_df - dist_ab - dist_cd - dist_ef, [(i + 1, k), (j + 1, k)]),  # Double reconnection
            (dist_ae + dist_bf + dist_ac + dist_bd - dist_ab - dist_cd - dist_ef, [(i + 1, j), (i + 1, k)]),  # Double reconnection
        ]
        
    
    def egal(self, other):
        return self.chemin == other.chemin