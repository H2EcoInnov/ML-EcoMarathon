import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Charger les données CSV
df = pd.read_excel('Documents/DEV/H2/Fay_Race_Track/Fay_de_Bretagne.xlsx')


# Afficher les premières lignes du DataFrame pour vérifier la lecture
print(df.head())


# Nettoyer les noms de colonnes pour éviter les espaces indésirables
df.columns = df.columns.str.strip()


# Extraire les colonnes pertinentes
distance = df['Distance (m)']
x = df['x (m)']
y = df['y (m)']
z = df['z (m)']

# Calculer la pente (steepness)
distance = df['Distance (m)']
steepness = np.gradient(z, distance)

# Normaliser les valeurs de la pente entre 0 et 1
steepness_normalized = (steepness - steepness.min()) / (steepness.max() - steepness.min())

# Choisir une largeur pour le circuit
width = 10  # par exemple, 10 mètres

# Créer une liste pour stocker les sommets des rectangles
verts = []
colors = []

# Parcourir les points pour créer des rectangles
for i in range(len(x) - 1):
    # Points du segment
    p1 = np.array([x[i], y[i], z[i]])
    p2 = np.array([x[i + 1], y[i + 1], z[i + 1]])

    # Calculer le vecteur de direction
    direction = p2 - p1
    direction /= np.linalg.norm(direction)  # Normaliser le vecteur

    # Calculer le vecteur normal (perpendiculaire)
    normal = np.array([-direction[1], direction[0], 0])  # On ignore la composante z

    # Points du rectangle (surface)
    p1_left = p1 + normal * (width / 2)
    p1_right = p1 - normal * (width / 2)
    p2_left = p2 + normal * (width / 2)
    p2_right = p2 - normal * (width / 2)

    # Ajouter le rectangle aux sommets
    verts.append([p1_left, p1_right, p2_right, p2_left])

    # Ajouter la couleur correspondant à la pente
    color = plt.cm.viridis(steepness_normalized[i])  # Utiliser la colormap 'viridis'
    colors.append(color)

# Créer une figure pour la visualisation
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Ajouter les surfaces du circuit avec les couleurs
for i, v in enumerate(verts):
    poly = Poly3DCollection([v], facecolors=colors[i], linewidths=1, edgecolors='none')
    ax.add_collection3d(poly)

# Définir les limites des axes
ax.set_xlim([x.min(), x.max()])
ax.set_ylim([y.min(), y.max()])
ax.set_zlim(60,150)

# Définir les labels et le titre
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Track Surface')

ax.grid(False)
# Afficher la figure
plt.show()