from copy import deepcopy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Individu import Individu

class Groupes:
    l = 100
    k = 5
    mutation_rate = 0.01
    def __init__(self, l, k, mutation_rate=0.01):
        self.l = l
        self.k = k
        self.mutation_rate = mutation_rate
        self.villes = {}
        self.individus = []

    def copy(self):
        clone = Groupes(self.l, self.k, mutation_rate=self.mutation_rate)
        clone.villes = deepcopy(self.villes)
        clone.individus = []
        clone.generate_individus()
        return clone

    def add_Ville(self, nom, coord):
        self.villes[nom] = coord

    def generate_individus(self):
        noms_villes = list(self.villes.keys())
        attempts = 0
        while len(self.individus) < self.l and attempts < self.l * 10:
            chemin = noms_villes.copy()
            random.shuffle(chemin)
            indiv = Individu(self, chemin)
            if not any(indiv.egal(i) for i in self.individus):
                self.individus.append(indiv)
            attempts += 1

    def distance_between(self, ville1, ville2):
        x1, y1 = self.villes[ville1]
        x2, y2 = self.villes[ville2]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def generate_random_city(self, size):
        self.villes = {}
        self.individus = []
        noms_villes = [chr(i) for i in range(65, 65 + size)]
        for nom in noms_villes:
            coord = [np.random.rand(), np.random.rand()]
            self.add_Ville(nom, coord)

    def select_tournament(self):
        selected = random.sample(self.individus, self.k)
        selected.sort(key=lambda indiv: indiv.fitness, reverse=True)
        return selected[0]
    
    def pmx(self, parent1, parent2, cut_points=None):
        n = len(parent1)
        if cut_points is None:
            c1, c2 = sorted(random.sample(range(n), 2))
        else:
            c1, c2 = cut_points

        child1, child2 = [None] * n, [None] * n

        # Pré-calcul des positions pour accès O(1)
        pos1 = {gene: i for i, gene in enumerate(parent1)}
        pos2 = {gene: i for i, gene in enumerate(parent2)}

        # 1) Copier les segments centraux (demi-ouvert: [c1, c2))
        child1[c1:c2] = parent1[c1:c2]
        child2[c1:c2] = parent2[c1:c2]

        # 2) Compléter via les correspondances de positions
        # Pour child1, on injecte les gènes du segment de parent2 aux bonnes positions
        for i in range(c1, c2):
            gene = parent2[i]
            if gene not in child1:
                pos = i
                # Trouver une case libre en suivant les images positionnelles
                while child1[pos] is not None:
                    mapped_gene = parent1[pos]
                    pos = pos2[mapped_gene]
                child1[pos] = gene

        # Pour child2, symétriquement
        for i in range(c1, c2):
            gene = parent1[i]
            if gene not in child2:
                pos = i
                while child2[pos] is not None:
                    mapped_gene = parent2[pos]
                    pos = pos1[mapped_gene]
                child2[pos] = gene

        # 3) Remplir les emplacements restants avec l'autre parent en conservant l'ordre
        for i in range(n):
            if child1[i] is None:
                child1[i] = parent2[i]
            if child2[i] is None:
                child2[i] = parent1[i]

        return child1, child2
    
    def cx(self, parent1, parent2):
        n = len(parent1)
        child1, child2 = [None]*n, [None]*n
        visited = [False]*n
        start = 0

        while not all(visited):
            idx = start
            while not visited[idx]:
                child1[idx] = parent1[idx]
                child2[idx] = parent2[idx]
                visited[idx] = True
                idx = parent1.index(parent2[idx])

            if not all(visited):
                start = visited.index(False)
                idx = start
                while not visited[idx]:
                    child1[idx] = parent2[idx]
                    child2[idx] = parent1[idx]
                    visited[idx] = True
                    idx = parent1.index(parent2[idx])

        return child1, child2
    
    def erx(self, parent1, parent2):
        def build_edge_map(p1, p2):
            edges = {c: set() for c in p1}
            for p in [p1, p2]:
                for i in range(len(p)):
                    left, right = p[i-1], p[(i+1) % len(p)]
                    edges[p[i]].update([left, right])
            return edges

        def make_child(edges):
            child = []
            current = random.choice(list(edges.keys()))
            while len(child) < len(edges):
                child.append(current)
                for e in edges.values():
                    e.discard(current)
                if edges[current]:
                    current = min(edges[current], key=lambda x: len(edges[x]))
                else:
                    remaining = [c for c in edges if c not in child]
                    if remaining:
                        current = random.choice(remaining)
            return child

        edges = build_edge_map(parent1, parent2)
        # Utiliser une copie des arêtes pour chaque enfant afin d'éviter les effets de bord
        edges_copy = {k: set(v) for k, v in edges.items()}
        return make_child(edges), make_child(edges_copy)

    def hx(self, parent1, parent2, distance_matrix):
        n = len(parent1)

        # Pré-calcul des positions pour accès O(1)
        pos1 = {city: i for i, city in enumerate(parent1)}
        pos2 = {city: i for i, city in enumerate(parent2)}

        def nearest_from(current, candidates):
            return min(candidates, key=lambda c: distance_matrix[current][c])

        def make_child():
            child = []
            used = set()
            current = random.choice(parent1)
            while len(child) < n:
                child.append(current)
                used.add(current)

                # Collecter voisins (prev et next) des deux parents
                cand = []
                i1 = pos1[current]
                i2 = pos2[current]
                neighs = [
                    parent1[(i1 - 1) % n], parent1[(i1 + 1) % n],
                    parent2[(i2 - 1) % n], parent2[(i2 + 1) % n],
                ]
                for v in neighs:
                    if v not in used:
                        cand.append(v)

                if cand:
                    current = nearest_from(current, cand)
                else:
                    remaining = [c for c in parent1 if c not in used]
                    if remaining:
                        current = nearest_from(current, remaining)
            return child

        return make_child(), make_child()

    
    def ox(self, parent1, parent2):
        n = len(self.villes)
        cut1, cut2 = sorted(random.sample(range(n), 2))

        # création des enfants
        enfant1 = [None] * n
        enfant2 = [None] * n

        # 2. échange des segments
        enfant1[cut1:cut2+1] = parent2[cut1:cut2+1]
        enfant2[cut1:cut2+1] = parent1[cut1:cut2+1]

        # 3. remplir tout en évitant les duplications (laisser a null)
        segment1 = set(enfant1[cut1:cut2+1])
        segment2 = set(enfant2[cut1:cut2+1])

        for i in range(n):
            if not (cut1 <= i <= cut2):
                if parent1[i] not in segment1:
                    enfant1[i] = parent1[i]
                if parent2[i] not in segment2:
                    enfant2[i] = parent2[i]

        # 4. villes manquantes
        missing1 = [v for v in self.villes if v not in enfant1]
        missing2 = [v for v in self.villes if v not in enfant2]

        random.shuffle(missing1)
        random.shuffle(missing2)

        # 5. remplir les trous
        for i in range(n):
            if enfant1[i] is None:
                enfant1[i] = missing1.pop()
            if enfant2[i] is None:
                enfant2[i] = missing2.pop()
        
        return enfant1, enfant2

    def croisement(self, method='ox', mutation='2-opt'):
        """Effectue le croisement entre deux individus"""
        parent1 = self.select_tournament()
        parent2 = self.select_tournament()
        while parent1.egal(parent2):
            parent2 = self.select_tournament()

        match method:
            case 'pmx':
                enfant1, enfant2 = self.pmx(parent1.chemin, parent2.chemin)
            case 'cx':
                enfant1, enfant2 = self.cx(parent1.chemin, parent2.chemin)
            case 'erx':
                enfant1, enfant2 = self.erx(parent1.chemin, parent2.chemin)
            case 'ox':
                enfant1, enfant2 = self.ox(parent1.chemin, parent2.chemin)
            case 'hx':
                distance_matrix = {v1: {v2: self.distance_between(v1, v2) for v2 in self.villes} for v1 in self.villes}
                enfant1, enfant2 = self.hx(parent1.chemin, parent2.chemin, distance_matrix)

        enfant1 = Individu(self, enfant1)
        enfant2 = Individu(self, enfant2)

        # mutation
        if np.random.rand() < self.mutation_rate:
            match mutation:
                case '2-opt':
                    enfant1.mutation_2opt()
                case '2-opt alt':
                    enfant1.mutation_2opt_alternative()
                case '3-opt':
                    enfant1.mutation_3opt()
        if np.random.rand() < self.mutation_rate:
            match mutation:
                case '2-opt':
                    enfant2.mutation_2opt()
                case '2-opt alt':
                    enfant2.mutation_2opt_alternative()
                case '3-opt':
                    enfant2.mutation_3opt()

        if not any(enfant1.egal(i) for i in self.individus):
            self.individus.append(enfant1)

        if not any(enfant2.egal(i) for i in self.individus):
            self.individus.append(enfant2)

        if (self.l != len(self.individus)):
            for i in range(len(self.individus) - self.l ):
                self.individus.pop(self.individus.index(min(self.individus, key=lambda indiv: indiv.fitness)))
        
        return enfant1, enfant2
    
    def generate_circle_city(self, size):
        self.villes = {}
        self.individus = []
        noms_villes = [chr(i) for i in range(65, 65 + size)]
        angle_step = 2 * np.pi / size
        for i, nom in enumerate(noms_villes):
            angle = i * angle_step
            coord = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]
            self.add_Ville(nom, coord)
    
    def plot_result(self, meilleur_individu, title="Meilleur chemin trouvé", display=True):
        """Affiche la carte 2D avec les villes et le meilleur chemin"""
        plt.figure(figsize=(10, 8))
        
        # Extraire les coordonnées des villes
        villes_coords = list(self.villes.values())
        x_coords = [coord[0] for coord in villes_coords]
        y_coords = [coord[1] for coord in villes_coords]
        
        # Afficher toutes les villes comme points
        plt.scatter(x_coords, y_coords, c='red', s=100, zorder=5)
        
        # Ajouter les noms des villes
        for nom, coord in self.villes.items():
            plt.annotate(nom, (coord[0], coord[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Tracer le chemin du meilleur individu
        chemin_coords = [self.villes[ville] for ville in meilleur_individu.chemin]
        chemin_x = [coord[0] for coord in chemin_coords]
        chemin_y = [coord[1] for coord in chemin_coords]
        
        # Fermer le circuit en ajoutant le point de départ à la fin
        chemin_x.append(chemin_x[0])
        chemin_y.append(chemin_y[0])
        
        # Tracer le chemin
        plt.plot(chemin_x, chemin_y, 'b-', linewidth=2, alpha=0.7, label=f'Distance: {meilleur_individu.total_distance:.3f}')
        
        # Marquer le point de départ en vert
        start_coord = self.villes[meilleur_individu.chemin[0]]
        plt.scatter(start_coord[0], start_coord[1], c='green', s=150, marker='s', zorder=6, label='Départ')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        if display: plt.show()
    
    def plot_evolution(self, fitness_history, display=True):
        """Affiche l'évolution de la fitness au cours des générations"""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.title('Évolution de la fitness au cours des générations', fontsize=14, fontweight='bold')
        plt.xlabel('Génération', fontsize=12)
        plt.ylabel('Meilleure fitness', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if display: plt.show()
    

    def animate_evolution(self, generations=5000, method='ox'):
        """Crée une animation qui capture seulement quand le meilleur chemin change"""
        
        # Réinitialiser la population
        self.individus = []
        self.generate_individus()
        
        # Variables pour l'animation
        frames_data = []
        meilleur_precedent = None
        
        # Évolution et collecte des données
        for generation in range(generations):
            self.croisement(method=method)
            meilleur_actuel = max(self.individus, key=lambda indiv: indiv.fitness)
            
            # Vérifier si le meilleur chemin a changé
            if meilleur_precedent is None or not meilleur_actuel.egal(meilleur_precedent):
                print(f"Génération {generation}: Nouveau meilleur! Distance = {meilleur_actuel.total_distance:.3f}")
                
                # Stocker les données de cette frame
                frames_data.append({
                    'generation': generation,
                    'chemin': meilleur_actuel.chemin.copy(),
                    'distance': meilleur_actuel.total_distance
                })
                meilleur_precedent = meilleur_actuel
        
        # Créer l'animation après avoir collecté toutes les données
        if frames_data:
            nb_images = len(frames_data)
            interval = 2000 / nb_images  # 2000ms / nombre d'images
            print(f"Animation avec {nb_images} images, intervalle: {interval:.1f}ms")
            
            # Configuration de la figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Ajouter des copies de la dernière frame pour la pause de 2 secondes
            nb_frames_pause = int(2000 / interval)  # Nombre de frames pour 2 secondes
            total_frames = nb_images + nb_frames_pause
            
            def animate_frame_with_pause(frame_idx):
                # Si on est dans la période de pause, utiliser la dernière image
                if frame_idx >= nb_images:
                    frame_idx = nb_images - 1
                
                ax.clear()
                ax.set_xlim(-0.1, 1.1)
                ax.set_ylim(-0.1, 1.1)
                ax.set_aspect('equal')
                
                frame_data = frames_data[frame_idx]
                ax.set_title(f"Gen {frame_data['generation']} - Distance: {frame_data['distance']:.3f}")
                ax.grid(True, alpha=0.3)
                
                # Afficher les villes
                villes_coords = list(self.villes.values())
                x_coords = [coord[0] for coord in villes_coords]
                y_coords = [coord[1] for coord in villes_coords]
                ax.scatter(x_coords, y_coords, c='red', s=100, zorder=5)
                
                # Tracer le chemin
                chemin_coords = [self.villes[ville] for ville in frame_data['chemin']]
                chemin_x = [coord[0] for coord in chemin_coords]
                chemin_y = [coord[1] for coord in chemin_coords]
                chemin_x.append(chemin_x[0])  # Fermer le circuit
                chemin_y.append(chemin_y[0])
                
                ax.plot(chemin_x, chemin_y, 'b-', linewidth=2, alpha=0.7)
                
                # Point de départ
                start_coord = self.villes[frame_data['chemin'][0]]
                ax.scatter(start_coord[0], start_coord[1], c='green', s=150, marker='s', zorder=6)
            
            # Créer l'animation avec les frames de pause
            anim = animation.FuncAnimation(fig, animate_frame_with_pause, frames=total_frames, interval=interval, repeat=True)
            
            try:
                # Sauvegarder
                anim.save("evolution_tsp.gif", writer='pillow')
                print("Animation sauvegardée: evolution_tsp.gif")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde: {e}")
            
            return anim
        else:
            print("Aucune amélioration détectée!")
            return None
    
    def __str__(self):
        return f"Villes: {self.villes}\nIndividus: {[indiv.chemin for indiv in self.individus]}"
            
