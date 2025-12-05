# dashboard_comparison.py - Version avec simulations RALENTIES
from __future__ import annotations
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time
import networkx as nx
import sys
import os
from typing import List, Dict, Tuple

# Ajouter le chemin src pour les imports absolus
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, os.path.abspath(src_dir))

# Import des modules SoDistinct avec imports absolus
try:
    from sodistinct.core.models import ICModel
    from sodistinct.core.graph_wrapper import GraphWrapper
    from sodistinct.core.engine import run_simulation
    from sodistinct.orchestrator.parallel import run_batch_parallel
    CORE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Erreur import core: {e}")
    CORE_AVAILABLE = False

# Import optionnels des backends distribués
try:
    from sodistinct.distributed.dask_backend import run_batch_dask
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from sodistinct.distributed.ray_backend import run_batch_ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

st.set_page_config(page_title="SoDistinct - Comparaison Performances", layout="wide")

# ============================================================================
# Configuration
# ============================================================================

st.title("⚡ SoDistinct - Comparaison des Modes d'Exécution")
st.markdown("**Comparez visuellement les performances de tous les backends !**")

if not CORE_AVAILABLE:
    st.error("❌ Impossible de charger les modules core de SoDistinct")
    st.stop()

# ============================================================================
# Configuration avancée - VERSION AMÉLIORÉE
# ============================================================================

with st.sidebar.expander("⚙️ Paramètres Avancés - CRITIQUES"):
    st.write("**🚨 PROBLÈME : Les simulations sont trop rapides !**")
    st.write("**SOLUTION : On ralentit artificiellement les simulations**")
    
    workload_type = st.selectbox(
        "Type de charge de travail:",
        ["Démo rapide", "Test réaliste", "Benchmark intensif", "TEST LOURD (pour voir différence)"]
    )
    
    # Configurations BEAUCOUP plus lourdes
    if workload_type == "Démo rapide":
        reseau_type = "Petit (50 nœuds)"
        num_simulations = 20
        simulation_difficulty = "Facile"
    elif workload_type == "Test réaliste":
        reseau_type = "Moyen (200 nœuds)"
        num_simulations = 50
        simulation_difficulty = "Moyen"
    elif workload_type == "Benchmark intensif":
        reseau_type = "Grand (500 nœuds)"
        num_simulations = 100
        simulation_difficulty = "Difficile"
    else:  # TEST LOURD
        reseau_type = "Très grand (1000 nœuds)"
        num_simulations = 200
        simulation_difficulty = "Très difficile"
    
    st.warning(f"**Configuration:** {reseau_type}, {num_simulations} simulations")
    st.info("💡 Les simulations sont ralenties artificiellement pour voir la différence")

# Sidebar configuration principale
st.sidebar.header("🔧 Configuration du Test")

# Backends à tester
st.sidebar.header("🎯 Backends à Comparer")

backends_selection = {
    "Séquentiel": st.sidebar.checkbox("Séquentiel", True),
    "Multithreading": st.sidebar.checkbox("Multithreading", True),
    "Multiprocessing": st.sidebar.checkbox("Multiprocessing", True),
}

if DASK_AVAILABLE:
    backends_selection["Dask"] = st.sidebar.checkbox("Dask", True)
else:
    st.sidebar.info("ℹ️ Dask non disponible")

if RAY_AVAILABLE:
    backends_selection["Ray"] = st.sidebar.checkbox("Ray", True)
else:
    st.sidebar.info("ℹ️ Ray non disponible")

# ============================================================================
# Générateur de réseaux BEAUCOUP plus GROS
# ============================================================================

def creer_reseau_test(taille: str) -> GraphWrapper:
    """Crée un réseau de test selon la taille demandée - VERSION AGRANDIE"""
    if taille == "Petit (50 nœuds)":
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
    elif taille == "Moyen (200 nœuds)":
        G = nx.erdos_renyi_graph(200, 0.1, seed=42)
    elif taille == "Grand (500 nœuds)":
        G = nx.erdos_renyi_graph(500, 0.05, seed=42)
    else:  # Très grand
        G = nx.erdos_renyi_graph(1000, 0.02, seed=42)
    
    return GraphWrapper(G)

def generer_seeds_sets(graph: GraphWrapper, n_sets: int) -> List[List[int]]:
    """Génère des seed sets aléatoires pour les tests"""
    nodes = list(graph.unwrap().nodes())
    seed_sets = []
    
    for i in range(n_sets):
        # Seed sets plus grands pour des simulations plus longues
        size = min(5, max(2, len(nodes) // 40))
        seeds = nodes[:size]
        seed_sets.append(seeds)
    
    return seed_sets

# ============================================================================
# MODÈLE PERSONNALISÉ - SIMULATIONS RALENTIES
# ============================================================================

class SlowICModel(ICModel):
    """
    Version ralentie du modèle IC pour forcer les différences de performance.
    Ajoute des calculs inutiles pour ralentir les simulations.
    """
    
    def step(self, graph, state, params):
        # Appel normal au modèle parent
        new_state = super().step(graph, state, params)
        
        # ⚠️ AJOUT DE CALCULS INUTILES POUR RALENTIR ⚠️
        # Cela simule des calculs plus complexes dans une vraie application
        slowdown_factor = 10000  # Ajustez ce facteur si nécessaire
        
        for _ in range(slowdown_factor):
            # Calcul inutile pour consommer du CPU
            dummy_calculation = sum(i*i for i in range(1000))
        
        return new_state

# ============================================================================
# Fonctions de test OPTIMISÉES pour voir la différence
# ============================================================================

def tester_sequentiel(graph: GraphWrapper, seed_sets: List[List[int]]) -> Tuple[float, List[float]]:
    """Test séquentiel - VERSION RALENTIE"""
    model = SlowICModel()  # Utilise le modèle ralenti
    params = {"p": 0.1}
    
    times = []
    start_total = time.time()
    
    for i, seeds in enumerate(seed_sets):
        start_sim = time.time()
        result = run_simulation(model, graph, seeds, params, rng_seed=42+i)
        times.append(time.time() - start_sim)
    
    total_time = time.time() - start_total
    return total_time, times

def tester_multithreading(graph: GraphWrapper, seed_sets: List[List[int]]) -> Tuple[float, List[float]]:
    """Test multithreading - VERSION RALENTIE"""
    model = SlowICModel()  # Utilise le modèle ralenti
    params = {"p": 0.1}
    
    start_total = time.time()
    results = run_batch_parallel(
        model=model,
        graph=graph,
        seed_sets=seed_sets,
        params=params,
        max_workers=4,
        use_processes=False
    )
    total_time = time.time() - start_total
    
    return total_time, [total_time/len(seed_sets)] * len(seed_sets)

def tester_multiprocessing(graph: GraphWrapper, seed_sets: List[List[int]]) -> Tuple[float, List[float]]:
    """Test multiprocessing - VERSION RALENTIE"""
    model = SlowICModel()  # Utilise le modèle ralenti
    params = {"p": 0.1}
    
    start_total = time.time()
    results = run_batch_parallel(
        model=model,
        graph=graph,
        seed_sets=seed_sets,
        params=params,
        max_workers=4,
        use_processes=True
    )
    total_time = time.time() - start_total
    
    return total_time, [total_time/len(seed_sets)] * len(seed_sets)

def tester_dask(graph: GraphWrapper, seed_sets: List[List[int]]) -> Tuple[float, List[float]]:
    """Test Dask - VERSION RALENTIE"""
    if not DASK_AVAILABLE:
        return 0, []
        
    model = SlowICModel()  # Utilise le modèle ralenti
    params = {"p": 0.1}
    
    start_total = time.time()
    try:
        results = run_batch_dask(
            model=model,
            graph=graph,
            seed_sets=seed_sets,
            params=params,
            address=None,
            scatter_graph=True
        )
        total_time = time.time() - start_total
        
        return total_time, [total_time/len(seed_sets)] * len(seed_sets)
    except Exception as e:
        st.error(f"❌ Erreur Dask: {e}")
        return 0, []

def tester_ray(graph: GraphWrapper, seed_sets: List[List[int]]) -> Tuple[float, List[float]]:
    """Test Ray - VERSION RALENTIE"""
    if not RAY_AVAILABLE:
        return 0, []
        
    model = SlowICModel()  # Utilise le modèle ralenti
    params = {"p": 0.1}
    
    start_total = time.time()
    try:
        results = run_batch_ray(
            model=model,
            graph=graph,
            seed_sets=seed_sets,
            params=params,
            use_actors=False,
            num_actors=4
        )
        total_time = time.time() - start_total
        
        return total_time, [total_time/len(seed_sets)] * len(seed_sets)
    except Exception as e:
        st.error(f"❌ Erreur Ray: {e}")
        return 0, []

# ============================================================================
# Visualisations
# ============================================================================

def plot_comparaison_temps(total_times: Dict[str, float]):
    """Graphique comparatif des temps totaux"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_backends = {k: v for k, v in total_times.items() if v > 0}
    
    if not valid_backends:
        st.warning("Aucune donnée valide pour le graphique")
        return fig
    
    backends = list(valid_backends.keys())
    times = list(valid_backends.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(backends, times, color=colors[:len(backends)])
    ax.set_ylabel('Temps Total (secondes)')
    ax.set_title('Comparaison des Temps d\'Exécution Totaux')
    ax.grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_speedup(total_times: Dict[str, float]):
    """Graphique du speedup par rapport au séquentiel"""
    if "Séquentiel" not in total_times or total_times["Séquentiel"] <= 0:
        return None
        
    seq_time = total_times["Séquentiel"]
    speedups = {}
    
    for backend, time_val in total_times.items():
        if backend != "Séquentiel" and time_val > 0:
            speedups[backend] = seq_time / time_val
    
    if not speedups:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    backends = list(speedups.keys())
    speeds = list(speedups.values())
    
    colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(backends, speeds, color=colors[:len(backends)])
    ax.set_ylabel('Speedup (x fois plus rapide)')
    ax.set_title('Speedup par Rapport au Mode Séquentiel')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Ligne de base (séquentiel)')
    ax.grid(True, alpha=0.3)
    
    for bar, speed in zip(bars, speeds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{speed:.1f}x', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    return fig

# ============================================================================
# Interface principale
# ============================================================================

if st.sidebar.button("🚀 Lancer la Comparaison", type="primary"):
    
    # Afficher un avertissement sur le ralentissement
    st.warning("""
    ⚠️ **ATTENTION : Simulations ralenties artificiellement**
    
    Les simulations contiennent des calculs supplémentaires pour :
    - Forcer les différences de performance entre backends
    - Rendre l'overhead des backends parallèles négligeable
    - Vous permettre de voir la VRAIE différence
    """)
    
    # Préparation
    with st.spinner("Préparation des données de test (réseaux plus gros)..."):
        graph = creer_reseau_test(reseau_type)
        seed_sets = generer_seeds_sets(graph, num_simulations)
    
    st.header("📊 Résultats de la Comparaison")
    
    # Métriques du réseau
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nœuds", graph.number_of_nodes())
    with col2:
        st.metric("Arêtes", graph.number_of_edges())
    with col3:
        st.metric("Simulations", num_simulations)
    with col4:
        st.metric("Backends testés", sum(backends_selection.values()))
    
    # Tests des backends sélectionnés
    total_times = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    backends_to_test = [b for b, selected in backends_selection.items() if selected]
    
    for i, backend in enumerate(backends_to_test):
        status_text.text(f"Test en cours : {backend}...")
        progress_bar.progress((i) / len(backends_to_test))
        
        try:
            if backend == "Séquentiel":
                total_time, times = tester_sequentiel(graph, seed_sets)
            elif backend == "Multithreading":
                total_time, times = tester_multithreading(graph, seed_sets)
            elif backend == "Multiprocessing":
                total_time, times = tester_multiprocessing(graph, seed_sets)
            elif backend == "Dask":
                total_time, times = tester_dask(graph, seed_sets)
            elif backend == "Ray":
                total_time, times = tester_ray(graph, seed_sets)
            else:
                continue
                
            total_times[backend] = total_time
            st.sidebar.success(f"✅ {backend}: {total_time:.2f}s")
            
        except Exception as e:
            st.error(f"❌ Erreur avec {backend}: {e}")
            total_times[backend] = 0
    
    progress_bar.progress(1.0)
    status_text.text("✅ Tous les tests sont terminés !")
    
    # Affichage des résultats
    st.subheader("📈 Graphiques de Comparaison")
    
    # Tableau des résultats
    st.write("### 📋 Tableau des Performances")
    df_results = pd.DataFrame([
        {
            "Backend": backend,
            "Temps Total (s)": f"{time:.2f}",
            "Temps Moyen par Sim (s)": f"{time/num_simulations:.4f}",
            "Simulations par Seconde": f"{num_simulations/time:.2f}" if time > 0 else "N/A"
        }
        for backend, time in total_times.items() if time > 0
    ])
    st.dataframe(df_results, use_container_width=True)
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temps = plot_comparaison_temps(total_times)
        st.pyplot(fig_temps)
    
    with col2:
        fig_speedup = plot_speedup(total_times)
        if fig_speedup:
            st.pyplot(fig_speedup)
        else:
            st.info("Speedup non disponible")
    
    # Analyse CRITIQUE des résultats
    st.subheader("🔍 Analyse CRITIQUE des Résultats")
    
    if "Séquentiel" in total_times and total_times["Séquentiel"] > 0:
        seq_time = total_times["Séquentiel"]
        
        # Vérification : Est-ce que les backends parallèles gagnent ENFIN ?
        parallel_wins = any(
            backend != "Séquentiel" and time_val > 0 and seq_time / time_val > 1.1
            for backend, time_val in total_times.items()
        )
        
        if parallel_wins:
            st.success("🎉 **ENFIN ! Vous voyez la différence !** Les backends parallèles sont plus rapides.")
        else:
            st.error("""
            ❌ **PROBLÈME PERSISTE** : Le séquentiel gagne toujours !
            
            **Causes possibles :**
            1. Votre CPU est très rapide
            2. Le ralentissement n'est pas suffisant
            3. Problème de configuration des backends parallèles
            
            **Solution :** Essayez le test "TEST LOURD"
            """)
        
        st.write("**Détail des performances :**")
        for backend, time_val in total_times.items():
            if backend != "Séquentiel" and time_val > 0:
                speedup = seq_time / time_val
                if speedup > 1.5:
                    st.success(f"✅ **{backend}**: {speedup:.1f}x plus rapide - PARFAIT !")
                elif speedup > 1.1:
                    st.info(f"ℹ️ **{backend}**: {speedup:.1f}x plus rapide - Bon")
                elif speedup > 0.9:
                    st.warning(f"⚠️ **{backend}**: {speedup:.1f}x - Presque pareil")
                else:
                    st.error(f"❌ **{backend}**: {speedup:.1f}x - Plus lent !")

else:
    # Écran d'accueil avec explication du problème
    st.error("""
    🚨 **PROBLÈME IDENTIFIÉ : Simulations trop rapides**
    
    **Pourquoi le séquentiel gagne toujours ?**
    - Les simulations réseau sont TRÈS rapides (millisecondes)
    - L'overhead des backends parallèles est plus grand que le temps gagné
    - Même avec 200 nœuds, chaque simulation est instantanée
    
    **SOLUTION IMPLÉMENTÉE :**
    - ✅ Réseaux BEAUCOUP plus gros (jusqu'à 1000 nœuds)
    - ✅ Modèle de simulation RALENTI artificiellement
    - ✅ Calculs supplémentaires pour forcer les différences
    
    **Test recommandé :** "TEST LOURD" dans les paramètres avancés
    """)

# ============================================================================
# Section debugging
# ============================================================================

with st.expander("🐛 Debugging - Pour les développeurs"):
    st.write("""
    **Si le séquentiel gagne toujours, voici les solutions :**
    
    1. **Augmenter le facteur de ralentissement** :
    ```python
    slowdown_factor = 50000  # Au lieu de 10000
    ```
    
    2. **Vérifier que les backends parallèles fonctionnent** :
    ```python
    # Test simple
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(lambda x: x*x, range(10)))
    print("Multiprocessing fonctionne:", results)
    ```
    
    3. **Vérifier les temps individuels** :
    ```python
    # Dans tester_sequentiel, afficher les temps
    print("Temps séquentiel:", times)
    ```
    """)