from __future__ import annotations
import streamlit as st
import matplotlib.pyplot as plt
import tempfile, os, time
from typing import List, Dict
import networkx as nx
import pandas as pd

# Import SoDistinct
from sodistinct.core.models import ICModel, LTModel, SIModel, SIRModel
from sodistinct.core.engine import run_simulation
from sodistinct.core.graph_wrapper import GraphWrapper
from sodistinct.io.loader import load_graph

st.set_page_config(page_title="SoDistinct - Temps Réel", layout="wide")

# ============================================================================
# Fonctions de visualisation temps réel
# ============================================================================

def visualiser_etape(graph: GraphWrapper, timeline_step: Dict, step_num: int):
    """Visualise une étape spécifique de la propagation"""
    G = graph.unwrap()
    pos = nx.spring_layout(G, seed=42)  # Position stable
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Réseau avec propagation
    active_nodes = set(timeline_step["active"])
    new_active_nodes = set(timeline_step["new_active"])
    
    # Tous les nœuds
    all_nodes = list(G.nodes())
    node_colors = []
    for node in all_nodes:
        if node in new_active_nodes:
            node_colors.append('red')  # Nouveaux activés - Rouge vif
        elif node in active_nodes:
            node_colors.append('orange')  # Déjà activés - Orange
        else:
            node_colors.append('lightblue')  # Pas encore activés - Bleu
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=300, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
    
    ax1.set_title(f"🔄 Étape {step_num}\n{len(active_nodes)} nœuds activés", 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Légende
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=10, label='Non activé'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                  markersize=10, label='Déjà activé'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                  markersize=10, label='Nouvellement activé')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Graphique 2: Courbe de progression
    steps = list(range(step_num + 1))
    cumulative = []
    for i in range(step_num + 1):
        cumulative.append(len(timeline_step["active"]))
    
    ax2.plot(steps, cumulative, 'g-', linewidth=3, marker='o', markersize=6)
    ax2.set_xlabel('Étape')
    ax2.set_ylabel('Nœuds Activés (cumul)')
    ax2.set_title('Progression de la Diffusion', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, len(G.nodes()) + 1)
    
    plt.tight_layout()
    return fig

def animer_propagation(graph: GraphWrapper, result, speed: float = 1.0):
    """Animation complète de la propagation"""
    st.subheader("🎬 Animation en Temps Réel")
    
    # Placeholder pour l'animation
    animation_placeholder = st.empty()
    stats_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Statistiques initiales
    total_nodes = len(graph.nodes())
    total_steps = len(result.timeline)
    
    # Animation étape par étape
    for step_num, timeline_step in enumerate(result.timeline):
        with animation_placeholder.container():
            # Visualisation de l'étape actuelle
            fig = visualiser_etape(graph, timeline_step, step_num)
            st.pyplot(fig)
            plt.close()
        
        # Mise à jour des statistiques
        with stats_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            active_count = len(timeline_step["active"])
            new_active_count = len(timeline_step["new_active"])
            
            with col1:
                st.metric("Étape", f"{step_num}/{total_steps-1}")
            with col2:
                st.metric("Activés", f"{active_count}/{total_nodes}")
            with col3:
                st.metric("Nouveaux", new_active_count)
            with col4:
                coverage = (active_count / total_nodes) * 100
                st.metric("Couverture", f"{coverage:.1f}%")
        
        # Barre de progression
        progress = (step_num + 1) / total_steps
        progress_placeholder.progress(
            progress, 
            text=f"Progression: {progress:.1%}"
        )
        
        # Pause contrôlable
        time.sleep(2.0 / speed)  # Plus speed est grand, plus c'est rapide
    
    # Animation terminée
    st.success("✅ **Animation terminée !**")
    
    # Résumé final
    final_active = len(result.timeline[-1]["active"])
    st.info(f"""
    **Résumé de la simulation:**
    - 🎯 **Couverture finale:** {final_active}/{total_nodes} nœuds ({final_active/total_nodes*100:.1f}%)
    - ⏱️ **Étapes nécessaires:** {total_steps}
    - 🚀 **Efficacité:** {'Excellent' if final_active/total_nodes > 0.8 else 'Bon' if final_active/total_nodes > 0.5 else 'Faible'}
    """)

# ============================================================================
# Interface Streamlit
# ============================================================================

st.title("🎯 SoDistinct - Visualisation Temps Réel")
st.markdown("**Observez la propagation d'information ÉTAPE PAR ÉTAPE en temps réel !**")

# ------------------ Sidebar ------------------
st.sidebar.header("1️⃣ Charger un graphe")

# Option: Fichier upload ou réseau prédéfini
option_graphe = st.sidebar.radio("Source du graphe:", 
                                ["Réseau prédéfini", "Upload fichier"])

graph: GraphWrapper | None = None

if option_graphe == "Upload fichier":
    uploaded_file = st.sidebar.file_uploader(
        "Fichier (.txt, .edgelist)", 
        type=["txt", "edgelist"]
    )
    
    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            graph = load_graph(tmp_path)
            g = graph.unwrap()
            st.sidebar.success(f"✅ {g.number_of_nodes()} nœuds, {g.number_of_edges()} arêtes")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur: {e}")
else:
    # Réseaux prédéfinis pour démonstration
    reseau_choisi = st.sidebar.selectbox(
        "Réseau prédéfini:",
        ["Petit réseau social", "Réseau en étoile", "Réseau communautaire"]
    )
    
    try:
        if reseau_choisi == "Petit réseau social":
            G = nx.Graph()
            G.add_edges_from([(0,1), (1,2), (2,3), (3,4), (0,4), (1,3), (2,4), (0,2)])
        elif reseau_choisi == "Réseau en étoile":
            G = nx.star_graph(8)  # Centre 0, branches 1-8
        else:  # Réseau communautaire
            G = nx.connected_caveman_graph(3, 4)  # 3 communautés de 4 personnes
        
        graph = GraphWrapper(G)
        st.sidebar.success(f"✅ {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur création réseau: {e}")

# ------------------ Configuration simulation ------------------
if graph is not None:
    st.sidebar.header("2️⃣ Configuration Simulation")
    
    # Modèle
    model_choice = st.sidebar.selectbox("Modèle de diffusion", ["IC", "SI", "SIR", "LT"])
    
    # Paramètres avec valeurs par défaut intelligentes
    default_params = {
        "IC": {"p": 0.3},
        "SI": {"transmission_rate": 0.2}, 
        "SIR": {"transmission_rate": 0.2, "recovery_rate": 0.1},
        "LT": {"threshold": 0.2}
    }
    
    params = {}
    for param_name, default_value in default_params[model_choice].items():
        if param_name == "p":
            params[param_name] = st.sidebar.slider("Probabilité transmission", 0.05, 1.0, default_value, 0.05)
        elif "rate" in param_name:
            params[param_name] = st.sidebar.slider(param_name.replace("_", " ").title(), 0.01, 1.0, default_value, 0.01)
        else:
            params[param_name] = st.sidebar.slider("Seuil", 0.05, 1.0, default_value, 0.05)
    
    # Seed set
    st.sidebar.header("3️⃣ Point de Départ")
    nodes_list = list(graph.unwrap().nodes())
    seed_nodes = st.sidebar.multiselect(
        "Qui commence à parler? (Seed set)",
        options=nodes_list,
        default=nodes_list[:1] if nodes_list else [],
        help="Sélectionnez les nœuds qui ont l'information au départ"
    )
    
    # Contrôles animation
    st.sidebar.header("4️⃣ Contrôles Animation")
    animation_speed = st.sidebar.slider("Vitesse animation", 0.5, 3.0, 1.0, 0.5)
    rng_seed = st.sidebar.number_input("Seed aléatoire", value=42)

# ============================================================================
# Zone principale
# ============================================================================

if graph is None:
    st.info("""
    👋 **Bienvenue dans SoDistinct Temps Réel !**
    
    Pour commencer:
    1. **Choisissez un réseau** dans la sidebar (prédéfini ou upload)
    2. **Configurez** la simulation  
    3. **Lancez l'animation** pour voir la propagation en direct !
    
    *Conseil: Commencez avec "Petit réseau social" pour une démonstration rapide.*
    """)
    st.stop()

# Affichage du réseau initial
st.header("📊 Réseau Initial")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nœuds", len(graph.nodes()))
with col2:
    st.metric("Connexions", len(graph.edges()))
with col3:
    st.metric("Densité", f"{nx.density(graph.unwrap()):.3f}")

# Visualisation statique du réseau initial
fig_init, ax = plt.subplots(figsize=(10, 8))
G_init = graph.unwrap()
pos_init = nx.spring_layout(G_init, seed=42)
nx.draw_networkx_nodes(G_init, pos_init, node_color='lightblue', node_size=300)
nx.draw_networkx_edges(G_init, pos_init, alpha=0.6)
nx.draw_networkx_labels(G_init, pos_init, font_size=8)
ax.set_title("Réseau Initial - Prêt pour la diffusion")
ax.axis('off')
st.pyplot(fig_init)
plt.close()

# Bouton de lancement
if st.button("🎬 Lancer l'Animation Temps Réel", type="primary", use_container_width=True):
    if not seed_nodes:
        st.error("❌ Veuillez sélectionner au moins un nœud de départ!")
        st.stop()
    
    # Simulation
    model_map = {"IC": ICModel, "SI": SIModel, "SIR": SIRModel, "LT": LTModel}
    model = model_map[model_choice]()
    
    with st.spinner(f'🚀 Simulation en cours avec modèle {model_choice}...'):
        result = run_simulation(
            model=model, 
            graph=graph, 
            seed_set=seed_nodes, 
            params=params, 
            rng_seed=rng_seed
        )
    
    # Animation
    animer_propagation(graph, result, animation_speed)

# Section éducative
st.markdown("---")
st.header("🎓 Guide d'Observation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👀 Que regarder pendant l'animation:")
    st.markdown("""
    - 🔴 **Rouge**: Nœuds **nouvellement** activés à cette étape
    - 🟠 **Orange**: Nœuds **déjà** activés aux étapes précédentes  
    - 🔵 **Bleu**: Nœuds **pas encore** atteints par l'information
    - 📈 **Courbe verte**: Progression **cumulative** de la diffusion
    """)

with col2:
    st.subheader("🔍 Phénomènes à observer:")
    st.markdown("""
    - **Effet de cluster**: Les groupes d'amis s'activent ensemble
    - **Goulots d'étranglement**: Connexions critiques entre communautés
    - **Saturation**: Quand la diffusion ralentit/stop
    - **Influence des seeds**: Impact du point de départ choisi
    """)