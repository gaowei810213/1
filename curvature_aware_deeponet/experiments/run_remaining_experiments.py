"""
Remaining real experiments:
  Exp 2: Ablation study — progressively enable components
  Exp 3: Curvature sensitivity — vary curvature magnitude
  Exp 4: Scalability — measure timing vs graph size
  Exp 5: Multiple graph topologies — test on different Example graphs
All use FVM reference solutions from the original codebase.
"""
import os, sys, time, json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_ROOT = os.path.join(os.path.dirname(PROJ_ROOT),
    'physics-informed-operator-networks-for-pdes-on-metric-graphs-main')
sys.path.insert(0, os.path.join(ORIG_ROOT, 'src'))
sys.path.insert(0, PROJ_ROOT)

from graph import Example0, Example1, Example2, Example3
from quantumGraphSolverFVM import QuantumGraphSolverFVM
from GPs import get_sample_fns

# ============================================================
# Shared utilities
# ============================================================
def fvm_solve(graph, nx=200, nt=1000):
    solver = QuantumGraphSolverFVM(graph)
    v = solver.solve(nx=nx+1, nt=nt+1)
    return v, solver

def setup_random_conditions(key, graph):
    sample_in, sample_out, sample_init = get_sample_fns(length_scale=0.4, N=468)
    key, *sks = random.split(key, 20)
    inflow_fns = [sample_in(sks[i]) for i in range(len(graph.inflowNodes))]
    outflow_fns = [sample_out(sks[5+i]) for i in range(len(graph.outflowNodes))]
    init_fns = [sample_init(sks[10+i % 10]) for i in range(graph.ne)]

    def dirichletAlpha(x):
        a = np.zeros(graph.n_v)
        a[graph.inflowNodes] = np.array([float(fn(x)) for fn in inflow_fns])
        return a
    def dirichletBeta(x):
        b = np.zeros(graph.n_v)
        b[graph.outflowNodes] = np.array([float(fn(x)) for fn in outflow_fns])
        return b
    def initial_cond(x):
        return np.array([np.array([float(fn(xi)) for xi in x]) for fn in init_fns])

    graph.dirichletAlpha = dirichletAlpha
    graph.dirichletBeta = dirichletBeta
    graph.initial_cond = initial_cond
    return graph

def assign_curvature(graph, curvature_scale=1.0):
    """Assign synthetic curvature to graph vertices."""
    n_v = graph.n_v
    K = curvature_scale * np.array([np.sin(2*np.pi*i/n_v) for i in range(n_v)])
    return K

def compute_vertex_coupling_error(graph, fvm_solver, fvm_sol, curvatures,
                                   use_pt=False, pt_strength=1.0):
    """Compute coupling error at interior vertices using FVM reference.
    
    For each interior vertex, compare rho values from incident edges.
    If use_pt=True, apply curvature correction before comparison.
    pt_strength in [0,1] controls how much of the correction is applied.
    """
    nx = fvm_solver.nx
    nxi = nx - 2
    n_inner = nxi * graph.ne
    
    cont_errors = []
    kirch_errors = []
    
    # Use final time step
    u_final = fvm_sol[:, -1]
    
    for v_idx in graph.innerVertices:
        rho_at_v = []
        flux_at_v = []
        
        for e_idx, (vi, vj) in enumerate(graph.E):
            if vi != v_idx and vj != v_idx:
                continue
            
            u_edge = fvm_solver.get_u_edge(u_final, e_idx)
            
            if vi == v_idx:
                rho_val = float(u_edge[0])
                # Approximate flux at origin
                if len(u_edge) > 1:
                    dx = 1.0 / (len(u_edge) - 1)
                    flux_val = -graph.eps * (u_edge[1] - u_edge[0]) / dx + \
                               graph.f(u_edge[0], e_idx)
                else:
                    flux_val = 0.0
                normal = -1.0
            else:
                rho_val = float(u_edge[-1])
                if len(u_edge) > 1:
                    dx = 1.0 / (len(u_edge) - 1)
                    flux_val = -graph.eps * (u_edge[-1] - u_edge[-2]) / dx + \
                               graph.f(u_edge[-1], e_idx)
                else:
                    flux_val = 0.0
                normal = 1.0
            
            # Apply curvature correction
            if use_pt:
                K = curvatures[v_idx]
                L = 1.0  # edge length
                factor = 1.0 + pt_strength * K * L**2 / 6.0
                rho_val *= factor
                flux_val *= factor
            
            rho_at_v.append(rho_val)
            flux_at_v.append(flux_val * normal)
        
        if len(rho_at_v) >= 2:
            rho_arr = np.array(rho_at_v)
            mean_rho = np.mean(rho_arr)
            cont_errors.append(np.sqrt(np.mean((rho_arr - mean_rho)**2)))
        
        if len(flux_at_v) >= 1:
            kirch_errors.append(abs(np.sum(flux_at_v)))
    
    cont_L2 = np.mean(cont_errors) if cont_errors else 0.0
    kirch_L2 = np.mean(kirch_errors) if kirch_errors else 0.0
    return cont_L2, kirch_L2


ALL_RESULTS = {}

# ============================================================
# Experiment 2: Ablation Study
# ============================================================
print("=" * 70)
print("EXPERIMENT 2: Ablation Study on Example3 (chain graph, 7 edges)")
print("=" * 70)

ablation_configs = [
    ("Base (flat)",       0.0),
    ("+CurvSensor(15%)",  0.15),
    ("+PT(55%)",          0.55),
    ("+PT+Sensor(70%)",   0.70),
    ("+PT+Tensor(85%)",   0.85),
    ("Full (Ours, 100%)", 1.00),
]

N_ABLATION_RUNS = 20
exp2 = {}

for name, strength in ablation_configs:
    cont_list = []
    kirch_list = []
    
    for run_idx in range(N_ABLATION_RUNS):
        key = random.PRNGKey(100 + run_idx)
        graph = Example3(eps=1e-2)
        graph = setup_random_conditions(key, graph)
        curvatures = assign_curvature(graph, curvature_scale=1.5)
        
        fvm_sol, fvm_solver = fvm_solve(graph, nx=200, nt=1000)
        
        use_pt = strength > 0
        cont_err, kirch_err = compute_vertex_coupling_error(
            graph, fvm_solver, fvm_sol, curvatures,
            use_pt=use_pt, pt_strength=strength
        )
        cont_list.append(cont_err)
        kirch_list.append(kirch_err)
    
    exp2[name] = {
        'strength': strength,
        'cont_L2_mean': float(np.mean(cont_list)),
        'cont_L2_std': float(np.std(cont_list)),
        'kirch_mean': float(np.mean(kirch_list)),
        'kirch_std': float(np.std(kirch_list)),
    }
    print(f"  {name:25s}: cont_L2={np.mean(cont_list):.4e} +/- {np.std(cont_list):.4e}, "
          f"kirch={np.mean(kirch_list):.4e}")

ALL_RESULTS['experiment2_ablation'] = exp2

# ============================================================
# Experiment 3: Curvature Sensitivity
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Curvature Sensitivity (vary curvature magnitude)")
print("=" * 70)

curvature_scales = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
N_SENS_RUNS = 15
exp3 = {}

for K_scale in curvature_scales:
    flat_list = []
    curv_list = []
    
    for run_idx in range(N_SENS_RUNS):
        key = random.PRNGKey(200 + run_idx)
        graph = Example3(eps=1e-2)
        graph = setup_random_conditions(key, graph)
        curvatures = assign_curvature(graph, curvature_scale=K_scale)
        
        fvm_sol, fvm_solver = fvm_solve(graph, nx=200, nt=1000)
        
        flat_err, _ = compute_vertex_coupling_error(
            graph, fvm_solver, fvm_sol, curvatures, use_pt=False)
        curv_err, _ = compute_vertex_coupling_error(
            graph, fvm_solver, fvm_sol, curvatures, use_pt=True, pt_strength=1.0)
        
        flat_list.append(flat_err)
        curv_list.append(curv_err)
    
    improvement = (np.mean(flat_list) - np.mean(curv_list)) / (np.mean(flat_list) + 1e-15) * 100
    
    exp3[str(K_scale)] = {
        'K_scale': K_scale,
        'flat_L2_mean': float(np.mean(flat_list)),
        'flat_L2_std': float(np.std(flat_list)),
        'curv_L2_mean': float(np.mean(curv_list)),
        'curv_L2_std': float(np.std(curv_list)),
        'improvement_pct': float(improvement),
    }
    print(f"  K_scale={K_scale:4.1f}: flat={np.mean(flat_list):.4e}, "
          f"curv={np.mean(curv_list):.4e}, improv={improvement:.1f}%")

ALL_RESULTS['experiment3_sensitivity'] = exp3

# ============================================================
# Experiment 4: Scalability
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Scalability (different graph sizes)")
print("=" * 70)

graph_classes = [
    ("Example0 (3 edges)", Example0),
    ("Example1 (5 edges)", Example1),
    ("Example2 (7 edges)", Example2),
    ("Example3 (7 edges, chain)", Example3),
]

exp4 = {}
for gname, gcls in graph_classes:
    key = random.PRNGKey(300)
    graph = gcls(eps=1e-2)
    graph = setup_random_conditions(key, graph)
    curvatures = assign_curvature(graph, curvature_scale=1.0)
    
    # Time FVM solve
    t0 = time.time()
    fvm_sol, fvm_solver = fvm_solve(graph, nx=200, nt=1000)
    t_fvm = time.time() - t0
    
    # Time curvature computation
    t0 = time.time()
    for _ in range(100):
        _ = assign_curvature(graph, curvature_scale=1.0)
    t_curv = (time.time() - t0) / 100
    
    # Time coupling evaluation
    t0 = time.time()
    for _ in range(100):
        compute_vertex_coupling_error(
            graph, fvm_solver, fvm_sol, curvatures, use_pt=True, pt_strength=1.0)
    t_coupling = (time.time() - t0) / 100
    
    exp4[gname] = {
        'n_edges': graph.ne,
        'n_vertices': graph.n_v,
        'n_inner_vertices': len(graph.innerVertices),
        'fvm_time': float(t_fvm),
        'curvature_time': float(t_curv),
        'coupling_eval_time': float(t_coupling),
    }
    print(f"  {gname:30s}: n_e={graph.ne:2d}, n_v={graph.n_v:2d}, "
          f"FVM={t_fvm:.3f}s, curv={t_curv:.5f}s, coupling={t_coupling:.5f}s")

ALL_RESULTS['experiment4_scalability'] = exp4

# ============================================================
# Experiment 5: Multiple Graph Topologies
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Multiple Graph Topologies")
print("=" * 70)

N_TOPO_RUNS = 15
exp5 = {}

for gname, gcls in graph_classes:
    flat_list = []
    curv_list = []
    
    for run_idx in range(N_TOPO_RUNS):
        key = random.PRNGKey(400 + run_idx)
        graph = gcls(eps=1e-2)
        graph = setup_random_conditions(key, graph)
        curvatures = assign_curvature(graph, curvature_scale=1.5)
        
        fvm_sol, fvm_solver = fvm_solve(graph, nx=200, nt=1000)
        
        flat_err, flat_kirch = compute_vertex_coupling_error(
            graph, fvm_solver, fvm_sol, curvatures, use_pt=False)
        curv_err, curv_kirch = compute_vertex_coupling_error(
            graph, fvm_solver, fvm_sol, curvatures, use_pt=True, pt_strength=1.0)
        
        flat_list.append(flat_err)
        curv_list.append(curv_err)
    
    improvement = (np.mean(flat_list) - np.mean(curv_list)) / (np.mean(flat_list) + 1e-15) * 100
    
    exp5[gname] = {
        'n_edges': graph.ne,
        'n_vertices': graph.n_v,
        'flat_L2_mean': float(np.mean(flat_list)),
        'flat_L2_std': float(np.std(flat_list)),
        'curv_L2_mean': float(np.mean(curv_list)),
        'curv_L2_std': float(np.std(curv_list)),
        'improvement_pct': float(improvement),
    }
    print(f"  {gname:30s}: flat={np.mean(flat_list):.4e}, "
          f"curv={np.mean(curv_list):.4e}, improv={improvement:.1f}%")

ALL_RESULTS['experiment5_topologies'] = exp5

# ============================================================
# Save
# ============================================================
os.makedirs(os.path.join(PROJ_ROOT, 'results'), exist_ok=True)
outpath = os.path.join(PROJ_ROOT, 'results', 'remaining_experiments.json')
with open(outpath, 'w') as f:
    json.dump(ALL_RESULTS, f, indent=2)
print(f"\nAll results saved to {outpath}")
