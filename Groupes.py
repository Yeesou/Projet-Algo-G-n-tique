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
    
    def croisement(self):
        parent1 = self.select_tournament()
        parent2 = self.select_tournament()
        while parent1.egal(parent2):
            parent2 = self.select_tournament()

        n = len(self.villes)
        cut1, cut2 = sorted(random.sample(range(n), 2))

        # création des enfants
        enfant1 = [None] * n
        enfant2 = [None] * n

        # 2. échange des segments
        enfant1[cut1:cut2+1] = parent2.chemin[cut1:cut2+1]
        enfant2[cut1:cut2+1] = parent1.chemin[cut1:cut2+1]

        # 3. remplir tout en évitant les duplications (laisser a null)
        segment1 = set(enfant1[cut1:cut2+1])
        segment2 = set(enfant2[cut1:cut2+1])

        for i in range(n):
            if not (cut1 <= i <= cut2):
                if parent1.chemin[i] not in segment1:
                    enfant1[i] = parent1.chemin[i]
                if parent2.chemin[i] not in segment2:
                    enfant2[i] = parent2.chemin[i]

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

        # création des individus
        indiv1, indiv2 = Individu(self, enfant1), Individu(self, enfant2)

        # mutation 
        if np.random.rand() < self.mutation_rate:
            indiv1.mutation_2opt()
        if np.random.rand() < self.mutation_rate:
            indiv2.mutation_2opt()

        if not any(indiv1.egal(i) for i in self.individus):
            self.individus.append(indiv1)
            
        if not any(indiv2.egal(i) for i in self.individus):
            self.individus.append(indiv2)
        
        if (self.l != len(self.individus)):
            for i in range(len(self.individus) - self.l ):
                self.individus.pop(self.individus.index(min(self.individus, key=lambda indiv: indiv.fitness)))

        
        return indiv1, indiv2

    def generate_circle_city(self, size):
        self.villes = {}
        self.individus = []
        noms_villes = [chr(i) for i in range(65, 65 + size)]
        angle_step = 2 * np.pi / size
        for i, nom in enumerate(noms_villes):
            angle = i * angle_step
            coord = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]
            self.add_Ville(nom, coord)
    
    def plot_result(self, meilleur_individu, title="Meilleur chemin trouvé"):
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
        plt.show()
    
    def plot_evolution(self, fitness_history):
        """Affiche l'évolution de la fitness au cours des générations"""
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, 'b-', linewidth=2)
        plt.title('Évolution de la fitness au cours des générations', fontsize=14, fontweight='bold')
        plt.xlabel('Génération', fontsize=12)
        plt.ylabel('Meilleure fitness', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    

    def animate_evolution(self, generations=5000):
        """Crée une animation qui capture seulement quand le meilleur chemin change"""
        
        # Réinitialiser la population
        self.individus = []
        self.generate_individus()
        
        # Variables pour l'animation
        frames_data = []
        meilleur_precedent = None
        
        # Évolution et collecte des données
        for generation in range(generations):
            self.croisement()
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
            
