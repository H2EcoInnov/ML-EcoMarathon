import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

distance_arrive =50

STARTING = 2  # Valeur pour la ligne de départ
FINISHING = 3  # Valeur pour la ligne d'arrivée


# Helper function to draw perpendicular lines based on slope
def draw_perpendicular_line(track_map, x_idx, y_idx, slope, line_length, width,type_line, grid_size):
    if slope != 0:
        # Perpendicular slope
        perp_slope = -1 / slope
    else:
        # Special case for vertical or near vertical slopes
        perp_slope = np.inf

    # Depending on the slope, determine the delta for drawing the perpendicular line
    if np.isinf(perp_slope):
        dx, dy = 0, 1  # Vertical line
    elif perp_slope == 0:
        dx, dy = 1, 0  # Horizontal line
    else:
        dx = 1 / np.sqrt(1 + perp_slope**2)
        dy = perp_slope * dx

    # Extend the line along the perpendicular direction
    for i in range(-line_length, 2*line_length):
        new_x = int(x_idx + i * dx)
        new_y = int(y_idx + i * dy)
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            track_map[new_y - width:new_y + width + 1, new_x] = STARTING if type_line == 0 else FINISHING

def build_custom_track_from_points(x_points, y_points, grid_size, track_width, save_map=False):
    
    if track_width == 'auto':
        track_width = int(grid_size * 0.02)

    # Initialiser une matrice vide avec des 0 
    track = np.zeros((grid_size, grid_size))
    
    # Mapper les points (x_points, y_points) dans la matrice
    x_indices = np.interp(x_points, (min(x_points)-10*track_width, max(x_points)+10*track_width), (0, grid_size - 1)).astype(int)
    y_indices = np.interp(y_points, (min(y_points)-10*track_width, max(y_points)+10*track_width), (0, grid_size - 1)).astype(int)
    
    # Fonction pour ajouter de la largeur autour des points
    def add_track_width(track_map, x_idx, y_idx, width):
        for dx in range(-int(width), int(width) + 1):
            for dy in range(-int(width), int(width) + 1):
                new_x = x_idx + dx
                new_y = y_idx + dy
                if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                    track_map[new_y, new_x] = 1  # 1 représente la piste

    # Ajouter les points dans la matrice avec la largeur définie
    for i in range(len(x_indices)):
        add_track_width(track, x_indices[i], y_indices[i], track_width)

    # Calculer les pentes au début et à la fin
    slope_start = (y_indices[1] - y_indices[0]) / (x_indices[1] - x_indices[0]) if (x_indices[1] - x_indices[0]) != 0 else np.inf
    slope_finish = (y_indices[distance_arrive + 1] - y_indices[distance_arrive]) / (x_indices[distance_arrive + 1] - x_indices[distance_arrive]) if (x_indices[distance_arrive + 1] - x_indices[distance_arrive]) != 0 else np.inf


    # Ajouter une ligne de départ
    draw_perpendicular_line(track, x_indices[0], y_indices[0], slope_start, line_length=track_width, width=1, type_line=0,grid_size=grid_size)

    # Ajouter une ligne d'arrivée perpendiculaire
    finish_x_idx, finish_y_idx = x_indices[distance_arrive], y_indices[distance_arrive]
    draw_perpendicular_line(track, finish_x_idx, finish_y_idx, slope_finish, line_length=track_width, width=1, type_line=1,grid_size=grid_size)
    
    # Enregistrer la matrice dans un fichier .npy si besoin
    if save_map:
        with open('ML_RL_Fay/track_a.npy', 'wb') as f:
            np.save(f, track)
    
    return track

# Exemple d'utilisation
# Charger les données du fichier Excel contenant les coordonnées (x, y)
df = pd.read_excel('Fay_Race_Track/Fay_de_Bretagne.xlsx')

# Assigner les colonnes aux variables
x_points = df['x (m)'].values
y_points = df['y (m)'].values

# Construire la piste avec une largeur et une taille de grille
track_matrix = build_custom_track_from_points(x_points, y_points, grid_size=100, track_width='auto', save_map=True)

# Optionnel : Visualiser la carte du circuit
plt.imshow(track_matrix, cmap='plasma', origin='lower')
plt.title("Track Map with Width")


plt.show()