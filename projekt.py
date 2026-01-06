import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, adjusted_rand_score
from ember_to_owl import has_api_action, api_actions, load_actions

# funkcia na extrakciu dat podla zvoleneho typu datasetu
def extract_experimental_matrix(ember_jsonl_path, actions_json_path, target_count, dataset_type="random"):
    load_actions(actions_json_path)
    all_possible_actions = sorted(api_actions.keys())
    action_to_idx = {action: i for i, action in enumerate(all_possible_actions)}
    
    samples_data = []
    sample_labels = [] 
    seen_profiles = set()
    family_buffer = {} 

    print(f"\n[*] Prehľadávam EMBER pre typ datasetu: {dataset_type.upper()}")

    with open(ember_jsonl_path, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            if sample.get("label") != 1: continue
            
            family = sample.get("avclass", "unknown")
            if family == "unknown" or family == "": continue

            current_vector = np.zeros(len(all_possible_actions), dtype=int)
            for action_name in all_possible_actions:
                if has_api_action(sample["imports"], action_name):
                    current_vector[action_to_idx[action_name]] = 1
            
            profile = tuple(current_vector)
            if profile in seen_profiles:
                continue

            if family not in family_buffer:
                family_buffer[family] = []
            
            family_buffer[family].append(current_vector)
            seen_profiles.add(profile)

            if dataset_type == "balanced":
                limit = target_count // 5
                eligible_families = [fam for fam, data in family_buffer.items() if len(data) >= limit]
                if len(eligible_families) >= 5:
                    for fam in eligible_families[:5]:
                        samples_data.extend(family_buffer[fam][:limit])
                        sample_labels.extend([f"{fam}"] * limit)
                    break

            elif dataset_type == "dominant":
                dom_limit = target_count // 2
                dom_candidates = [fam for fam, data in family_buffer.items() if len(data) >= dom_limit]
                
                if len(dom_candidates) >= 1:
                    dom_fam = dom_candidates[0]
                    all_others = []
                    other_labels = []
                    for fam, data in family_buffer.items():
                        if fam != dom_fam:
                            for vec in data:
                                all_others.append(vec)
                                other_labels.append(f"{fam}")
                    
                    if len(all_others) >= (target_count - dom_limit):
                        samples_data.extend(family_buffer[dom_fam][:dom_limit])
                        sample_labels.extend([f"{dom_fam}"] * dom_limit)
                        
                        remaining_needed = target_count - dom_limit
                        samples_data.extend(all_others[:remaining_needed])
                        sample_labels.extend(other_labels[:remaining_needed])
                        break
            
            elif dataset_type == "random":
                if len(seen_profiles) >= target_count:
                    count = 0
                    for fam, data in family_buffer.items():
                        for vec in data:
                            if count < target_count:
                                samples_data.append(vec)
                                sample_labels.append(f"{fam}")
                                count += 1
                        if count >= target_count: break
                    break

    if not samples_data:
        print(f"[!] Chyba: Nepodarilo sa naplniť podmienky pre scenár {dataset_type}")
        return None, None, None

    final_family_counts = {}
    for label in sample_labels:
        final_family_counts[label] = final_family_counts.get(label, 0) + 1
    print(f"[+] Rozdelenie rodín: {final_family_counts}")
    
    return np.array(samples_data), all_possible_actions, sample_labels

# hlavny experimentalny cyklus
path = "train_features_1.jsonl"
actions_path = "actions.json"
datasets = ["balanced", "dominant", "random"]

base_res_dir = "vysledky_projektu"
os.makedirs(base_res_dir, exist_ok=True)

for scene in datasets:
    matrix, feature_names, sample_labels = extract_experimental_matrix(path, actions_path, 100, dataset_type=scene)
    if matrix is None: continue

    scene_dir = os.path.join(base_res_dir, scene)
    dendrogram_dir = os.path.join(scene_dir, "vsetky_konfiguracie")
    os.makedirs(dendrogram_dir, exist_ok=True)

    metrics = ['jaccard', 'sokalmichener', 'hamming', 'dice']
    methods = ['average', 'complete', 'single']

    best_c = -1
    best_config = None

    print(f"[*] Scenár {scene.upper()}: Validujem 12 kombinácií...")

    for metric in metrics:
        for method in methods:
            try:
                # hierarchicky clustering a cophenetic koeficient
                Z = linkage(matrix, method=method, metric=metric)
                max_d = max(Z[:, 2])
                c, _ = cophenet(Z, pdist(matrix, metric=metric))

                # výpočet ARI a Silhouette Score
                # odrežeme hierarchicke clusters v Z na n_clusters skupín
                cluster_assignments = fcluster(Z, t=0.5 * max_d, criterion='distance')
                
                # ARI - nakoľko zhluky súhlasia s rodinami (sample_labels)
                ari = adjusted_rand_score(sample_labels, cluster_assignments)
                
                # Silhouette - vnútorná konzistencia (čím vyššie, tým lepšie)
                # pre silhouette musíme použiť rovnakú metriku ako v linkage
                sil = silhouette_score(matrix, cluster_assignments, metric=metric)

                if c > best_c:
                    best_c = c
                    best_config = (metric, method, ari, sil)

                # ukladanie dendrogramu s rozšírenými info
                plt.figure(figsize=(15, 10))
                dendrogram(Z, color_threshold=0.5 * max_d,labels=sample_labels, leaf_rotation=90, leaf_font_size=8)
                m_name = "SMC" if metric == "sokalmichener" else metric.capitalize()
                
                info_text = f"Cophenetic: {c:.3f} | ARI: {ari:.3f} | Silhouette: {sil:.3f}"
                plt.title(f"Scenár: {scene.upper()} | {m_name} + {method.upper()}\n{info_text}")
                
                plt.tight_layout()
                plt.savefig(os.path.join(dendrogram_dir, f"{metric}_{method}.png"))
                plt.close()

            except Exception as e:
                print(f"[-] Chyba pri {metric}/{method}: {e}")

    # heatmapa pre matematicky najvernejšiu konfiguráciu dendrogramu (max C)
    if best_config:
        m, meth, final_ari, final_sil = best_config

        active_idx = np.where(matrix.sum(axis=0) > 0)[0]
        sub_matrix = matrix[:, active_idx]
        sub_features = [feature_names[i] for i in active_idx]

        g = sns.clustermap(sub_matrix, metric=m, method=meth, figsize=(22, 15), 
                           cmap="YlGnBu", xticklabels=sub_features, yticklabels=sample_labels)
        
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=9)
        g.fig.suptitle(f"HEATMAPA {scene.upper()}: {m}+{meth}\nARI: {final_ari:.3f} | Sil: {final_sil:.3f}", fontsize=16)
        
        plt.savefig(os.path.join(scene_dir, f"HEATMAPA_{scene}.png"))
        plt.close()

print("\nExperimenty hotové.")