import sys
import pathlib
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk

# Ensure src is in the python path if executed standalone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

from src.mo_module.benchmark import DEFAULT_BENCHMARK_CIRCUITS, analyze_results
from src.mo_module.benchmark.runner import BenchmarkRun, BenchmarkResultSet
from src.mo_module.optimizer import OptimizerConfig, optimize_layout
from src.mo_module.tuning import LayoutTuner, HyperparameterSpace
from src.qiskit_interface.backend_info import get_backend
from src.qiskit_interface.transpiler import transpile_circuit

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# --- Lógica de Qiskit envuelta (fuera de la clase para evitar problemas de pickling con ProcessPool) ---
def _run_mo_one(bc_name, circuit, seed, backend, config):
    run_cfg = OptimizerConfig(
        algorithm=config.algorithm,
        population_size=config.population_size,
        n_generations=config.n_generations,
        objectives=list(config.objectives),
        optimization_level=config.optimization_level,
        crossover_operator=config.crossover_operator,
        prob_crossover=config.prob_crossover,
        prob_swap_mutation=config.prob_swap_mutation,
        prob_replace_mutation=config.prob_replace_mutation,
        seed=seed,
        verbose=False,
    )
    try:
        opt = optimize_layout(circuit=circuit, backend=backend, config=run_cfg)
        return BenchmarkRun(circuit_name=bc_name, seed=seed, result=opt)
    except Exception as exc:
        return BenchmarkRun(circuit_name=bc_name, seed=seed, error=str(exc))

def _run_baseline_one(bc_name, circuit, seed, backend, optimization_level):
    try:
        tr = transpile_circuit(
            circuit=circuit,
            backend=backend,
            optimization_level=optimization_level,
            seed=seed,
        )
        tm = tr.transpiled_metrics
        return bc_name, seed, {'depth': float(tm.depth), 'cnot_count': float(tm.cnot_equivalent)}
    except Exception:
        return bc_name, seed, {'depth': float('inf'), 'cnot_count': float('inf')}

class BenchmarkGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Benchmark Multi-Objective Layout Optimizer")
        self.geometry("1400x900")
        
        # Grid Layout (1 row, 1 column)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main Master TabView for "Benchmark" and "Tuning"
        self.master_tabview = ctk.CTkTabview(self)
        self.master_tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.tab_benchmark = self.master_tabview.add("Modo Benchmark")
        self.tab_benchmark.grid_rowconfigure(0, weight=1)
        self.tab_benchmark.grid_columnconfigure(1, weight=1)
        
        self.tab_tuning = self.master_tabview.add("Modo Tuning (Optuna)")
        self.tab_tuning.grid_rowconfigure(0, weight=1)
        self.tab_tuning.grid_columnconfigure(1, weight=1)

        self._create_benchmark_ui()
        self._create_tuning_ui()
        
        # Benchmark State
        self.is_running = False
        self._last_results = None
        self._last_report = None
        self._last_baseline = None

        # Tuning State
        self.is_tuning = False
        self._tuner = None

    def _create_benchmark_ui(self):
        self._create_sidebar(self.tab_benchmark)
        self._create_main_frame(self.tab_benchmark)
        
        # Benchmark State
        self.is_running = False
        self._last_results = None
        self._last_report = None
        self._last_baseline = None

    def _create_sidebar(self, parent):
        self.sidebar_frame = ctk.CTkFrame(parent, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)

        logo_label = ctk.CTkLabel(self.sidebar_frame, text="Configuración", font=ctk.CTkFont(size=20, weight="bold"))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Circuitos (Checkboxes en un ScrollableFrame)
        self.circuits_label = ctk.CTkLabel(self.sidebar_frame, text="Circuitos:", anchor="w")
        self.circuits_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.circuits_frame = ctk.CTkScrollableFrame(self.sidebar_frame, height=120)
        self.circuits_frame.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        self.circuit_vars = {}
        for bc in DEFAULT_BENCHMARK_CIRCUITS:
            var = ctk.BooleanVar(value=True)
            self.circuit_vars[bc.name] = var
            cb = ctk.CTkCheckBox(self.circuits_frame, text=bc.name, variable=var)
            cb.pack(pady=2, anchor="w")

        # Semillas
        self.seeds_label = ctk.CTkLabel(self.sidebar_frame, text="Semillas: 10", anchor="w")
        self.seeds_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.seeds_slider = ctk.CTkSlider(self.sidebar_frame, from_=1, to=30, number_of_steps=29, command=self._update_seeds_label)
        self.seeds_slider.set(10)
        self.seeds_slider.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Backend
        self.backend_label = ctk.CTkLabel(self.sidebar_frame, text="Backend:", anchor="w")
        self.backend_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.backend_option = ctk.CTkOptionMenu(self.sidebar_frame, values=['fake_torino', 'fake_sherbrooke', 'fake_brisbane'])
        self.backend_option.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Algoritmo
        self.algo_label = ctk.CTkLabel(self.sidebar_frame, text="Algoritmo:", anchor="w")
        self.algo_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.algo_option = ctk.CTkOptionMenu(self.sidebar_frame, values=['nsga2', 'moead'])
        self.algo_option.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.swap_mut_label = ctk.CTkLabel(self.sidebar_frame, text="Swap Mutation: 0.3", anchor="w")
        self.swap_mut_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        self.swap_mut_option = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["0.1", "0.3", "0.5", "0.7"],
            command=self._update_swap_mut_label,
        )
        self.swap_mut_option.set("0.3")
        self.swap_mut_option.grid(row=10, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.replace_mut_label = ctk.CTkLabel(self.sidebar_frame, text="Replace Mutation: 0.7", anchor="w")
        self.replace_mut_label.grid(row=11, column=0, padx=20, pady=(10, 0), sticky="w")
        self.replace_mut_option = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["0.1", "0.3", "0.5", "0.7", "0.9"],
            command=self._update_replace_mut_label,
        )
        self.replace_mut_option.set("0.7")
        self.replace_mut_option.grid(row=12, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # Separator for second part
        # Población
        self.pop_label = ctk.CTkLabel(self.sidebar_frame, text="Población: 30", anchor="w")
        self.pop_label.grid(row=13, column=0, padx=20, pady=(10, 0), sticky="w")
        self.pop_slider = ctk.CTkSlider(self.sidebar_frame, from_=6, to=100, number_of_steps=94, command=self._update_pop_label)
        self.pop_slider.set(30)
        self.pop_slider.grid(row=14, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Generaciones
        self.gens_label = ctk.CTkLabel(self.sidebar_frame, text="Generaciones: 50", anchor="w")
        self.gens_label.grid(row=15, column=0, padx=20, pady=(10, 0), sticky="w")
        self.gens_slider = ctk.CTkSlider(self.sidebar_frame, from_=5, to=200, number_of_steps=195, command=self._update_gens_label)
        self.gens_slider.set(50)
        self.gens_slider.grid(row=16, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Workers
        n_cpus = os.cpu_count() or 1
        default_workers = max(1, n_cpus // 2)
        max_workers = max(default_workers * 2, 8)
        self.workers_label = ctk.CTkLabel(self.sidebar_frame, text=f"Workers: {default_workers} (Max {max_workers})", anchor="w")
        self.workers_label.grid(row=17, column=0, padx=20, pady=(10, 0), sticky="w")
        self.workers_slider = ctk.CTkSlider(self.sidebar_frame, from_=1, to=max_workers, number_of_steps=max_workers-1, command=self._update_workers_label)
        self.workers_slider.set(default_workers)
        self.workers_slider.grid(row=18, column=0, padx=20, pady=(0, 20), sticky="ew")

        # Run Button
        self.run_button = ctk.CTkButton(self.sidebar_frame, text="▶ Ejecutar Benchmark", command=self.start_benchmark)
        self.run_button.grid(row=19, column=0, padx=20, pady=20, sticky="ew")

    def _create_main_frame(self, parent):
        self.main_frame = ctk.CTkFrame(parent)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # TabView for Results / Plots
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.grid(row=1, column=0, padx=10, pady=(10, 10), sticky="nsew")
        
        self.tab_log = self.tabview.add("Progreso & Terminal")
        self.tab_summary = self.tabview.add("Resumen Estadístico")
        self.tab_plots = self.tabview.add("Gráficos")

        # -- Tab Progreso --
        self.tab_log.grid_rowconfigure(1, weight=1)
        self.tab_log.grid_columnconfigure(0, weight=1)
        
        self.progress_label = ctk.CTkLabel(self.tab_log, text="Esperando inicio...", font=ctk.CTkFont(weight="bold"))
        self.progress_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self.tab_log)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.progress_bar.set(0)
        
        self.terminal_text = ctk.CTkTextbox(self.tab_log, state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.terminal_text.grid(row=2, column=0, sticky="nsew")

        # -- Tab Resumen --
        self.tab_summary.grid_rowconfigure(0, weight=1)
        self.tab_summary.grid_columnconfigure(0, weight=1)
        self.summary_text = ctk.CTkTextbox(self.tab_summary, state="disabled", font=ctk.CTkFont(family="Consolas", size=12))
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        
        # -- Tab Plots --
        self.tab_plots.grid_rowconfigure(0, weight=1)
        self.tab_plots.grid_columnconfigure(0, weight=1)
        self.plot_frame = ctk.CTkFrame(self.tab_plots)
        self.plot_frame.grid(row=0, column=0, sticky="nsew")

    def _create_tuning_ui(self):
        # Frame lateral (Configuración)
        self.t_sidebar = ctk.CTkFrame(self.tab_tuning, width=350, corner_radius=0)
        self.t_sidebar.grid(row=0, column=0, sticky="nsew")
        self.t_sidebar.grid_rowconfigure(20, weight=1)

        logo_label = ctk.CTkLabel(self.t_sidebar, text="Configuración Tuning", font=ctk.CTkFont(size=18, weight="bold"))
        logo_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))

        # Circuito (solo 1 para tuning)
        ctk.CTkLabel(self.t_sidebar, text="Circuito:").grid(row=1, column=0, padx=10, sticky="w")
        circuit_names = [bc.name for bc in DEFAULT_BENCHMARK_CIRCUITS]
        self.t_circuit = ctk.CTkOptionMenu(self.t_sidebar, values=circuit_names)
        self.t_circuit.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Backend
        ctk.CTkLabel(self.t_sidebar, text="Backend:").grid(row=2, column=0, padx=10, sticky="w")
        self.t_backend = ctk.CTkOptionMenu(self.t_sidebar, values=['fake_torino', 'fake_sherbrooke', 'fake_brisbane'])
        self.t_backend.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Trials
        self.t_trials_label = ctk.CTkLabel(self.t_sidebar, text="Trials: 10")
        self.t_trials_label.grid(row=3, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        self.t_trials_slider = ctk.CTkSlider(self.t_sidebar, from_=5, to=60, number_of_steps=55, command=lambda v: self.t_trials_label.configure(text=f"Trials: {int(v)}"))
        self.t_trials_slider.set(10)
        self.t_trials_slider.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")

        # Seeds/Trial
        self.t_seeds_label = ctk.CTkLabel(self.t_sidebar, text="Seeds/Trial: 3")
        self.t_seeds_label.grid(row=5, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        self.t_seeds_slider = ctk.CTkSlider(self.t_sidebar, from_=1, to=10, number_of_steps=9, command=lambda v: self.t_seeds_label.configure(text=f"Seeds/Trial: {int(v)}"))
        self.t_seeds_slider.set(3)
        self.t_seeds_slider.grid(row=6, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")

        # Rangos Pop & Gen
        ctk.CTkLabel(self.t_sidebar, text="Rango Población:", font=ctk.CTkFont(weight="bold")).grid(row=7, column=0, columnspan=2, padx=10, pady=(10,0), sticky="w")
        
        frame_pop = ctk.CTkFrame(self.t_sidebar, fg_color="transparent")
        frame_pop.grid(row=8, column=0, columnspan=2, sticky="ew", padx=10)
        ctk.CTkLabel(frame_pop, text="Min:").pack(side="left")
        self.t_pop_min = ctk.CTkEntry(frame_pop, width=50)
        self.t_pop_min.insert(0, "20")
        self.t_pop_min.pack(side="left", padx=(5, 15))
        ctk.CTkLabel(frame_pop, text="Max:").pack(side="left")
        self.t_pop_max = ctk.CTkEntry(frame_pop, width=50)
        self.t_pop_max.insert(0, "80")
        self.t_pop_max.pack(side="left", padx=(5, 0))

        ctk.CTkLabel(self.t_sidebar, text="Rango Generaciones:", font=ctk.CTkFont(weight="bold")).grid(row=9, column=0, columnspan=2, padx=10, pady=(10,0), sticky="w")
        
        frame_gen = ctk.CTkFrame(self.t_sidebar, fg_color="transparent")
        frame_gen.grid(row=10, column=0, columnspan=2, sticky="ew", padx=10)
        ctk.CTkLabel(frame_gen, text="Min:").pack(side="left")
        self.t_gen_min = ctk.CTkEntry(frame_gen, width=50)
        self.t_gen_min.insert(0, "30")
        self.t_gen_min.pack(side="left", padx=(5, 15))
        ctk.CTkLabel(frame_gen, text="Max:").pack(side="left")
        self.t_gen_max = ctk.CTkEntry(frame_gen, width=50)
        self.t_gen_max.insert(0, "120")
        self.t_gen_max.pack(side="left", padx=(5, 0))

        ctk.CTkLabel(self.t_sidebar, text="Categorías Mutación Swap:", font=ctk.CTkFont(weight="bold")).grid(row=11, column=0, columnspan=2, padx=10, pady=(10,0), sticky="w")
        self.t_swap_choices = ctk.CTkEntry(self.t_sidebar)
        self.t_swap_choices.insert(0, "0.1,0.3,0.5,0.7")
        self.t_swap_choices.grid(row=12, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")

        ctk.CTkLabel(self.t_sidebar, text="Categorías Mutación Replace:", font=ctk.CTkFont(weight="bold")).grid(row=13, column=0, columnspan=2, padx=10, pady=(10,0), sticky="w")
        self.t_replace_choices = ctk.CTkEntry(self.t_sidebar)
        self.t_replace_choices.insert(0, "0.1,0.3,0.5,0.7,0.9")
        self.t_replace_choices.grid(row=14, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")

        # Checkboxes Objetivos
        ctk.CTkLabel(self.t_sidebar, text="Objetivos:", font=ctk.CTkFont(weight="bold")).grid(row=15, column=0, columnspan=2, padx=10, pady=(15,0), sticky="w")
        self.t_obj_depth = ctk.CTkCheckBox(self.t_sidebar, text="depth")
        self.t_obj_depth.grid(row=16, column=0, padx=20, sticky="w"); self.t_obj_depth.select()
        self.t_obj_cnot = ctk.CTkCheckBox(self.t_sidebar, text="cnot_count")
        self.t_obj_cnot.grid(row=17, column=0, padx=20, sticky="w"); self.t_obj_cnot.select()

        # Checkboxes Algoritmos
        ctk.CTkLabel(self.t_sidebar, text="Algoritmos Space:", font=ctk.CTkFont(weight="bold")).grid(row=18, column=0, columnspan=2, padx=10, pady=(10,0), sticky="w")
        self.t_alg_nsga2 = ctk.CTkCheckBox(self.t_sidebar, text="nsga2")
        self.t_alg_nsga2.grid(row=19, column=0, padx=20, sticky="w"); self.t_alg_nsga2.select()
        self.t_alg_moead = ctk.CTkCheckBox(self.t_sidebar, text="moead")
        self.t_alg_moead.grid(row=20, column=0, padx=20, sticky="w")

        # Workers (Tuning)
        n_cpus = os.cpu_count() or 1
        default_workers = max(1, n_cpus // 2)
        max_workers = max(default_workers * 2, 8)
        self.t_workers_label = ctk.CTkLabel(self.t_sidebar, text=f"Workers: {default_workers} (Max {max_workers})", anchor="w")
        self.t_workers_label.grid(row=21, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="w")
        self.t_workers_slider = ctk.CTkSlider(self.t_sidebar, from_=1, to=max_workers, number_of_steps=max_workers-1, command=lambda v: self.t_workers_label.configure(text=f"Workers: {int(v)} (Max {int(max_workers)})"))
        self.t_workers_slider.set(default_workers)
        self.t_workers_slider.grid(row=22, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        # Config Output (ReadOnly)
        self.t_best_config_btn = ctk.CTkButton(self.t_sidebar, text="Copiar a Benchmark", state="disabled", fg_color="green", command=self._copy_best_to_benchmark)
        self.t_best_config_btn.grid(row=23, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

        # Run Button
        self.t_run_button = ctk.CTkButton(self.t_sidebar, text="▶ Iniciar Tuning Optuna", command=self.start_tuning)
        self.t_run_button.grid(row=24, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # Main Frame Tuning
        self.t_main_frame = ctk.CTkFrame(self.tab_tuning)
        self.t_main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.t_main_frame.grid_rowconfigure(2, weight=1)
        self.t_main_frame.grid_columnconfigure(0, weight=1)

        self.t_progress_label = ctk.CTkLabel(self.t_main_frame, text="Esperando inicio...", font=ctk.CTkFont(weight="bold"))
        self.t_progress_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        
        self.t_terminal_text = ctk.CTkTextbox(self.t_main_frame, height=300, font=ctk.CTkFont(family="Consolas", size=12))
        self.t_terminal_text.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.t_terminal_text.configure(state="disabled")

        self.t_plot_frame = ctk.CTkFrame(self.t_main_frame)
        self.t_plot_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    def log(self, text):
        self.terminal_text.configure(state="normal")
        self.terminal_text.insert("end", text + "\n")
        self.terminal_text.see("end")
        self.terminal_text.configure(state="disabled")

    def t_log(self, text):
        self.t_terminal_text.configure(state="normal")
        self.t_terminal_text.insert("end", text + "\n")
        self.t_terminal_text.see("end")
        self.t_terminal_text.configure(state="disabled")
        self.terminal_text.configure(state="normal")
        self.terminal_text.insert("end", text + "\n")
        self.terminal_text.see("end")
        self.terminal_text.configure(state="disabled")

    def _update_seeds_label(self, value):
        self.seeds_label.configure(text=f"Semillas: {int(value)}")

    def _update_pop_label(self, value):
        self.pop_label.configure(text=f"Población: {int(value)}")

    def _update_swap_mut_label(self, value):
        self.swap_mut_label.configure(text=f"Swap Mutation: {value}")

    def _update_replace_mut_label(self, value):
        self.replace_mut_label.configure(text=f"Replace Mutation: {value}")

    def _update_gens_label(self, value):
        self.gens_label.configure(text=f"Generaciones: {int(value)}")

    def _update_workers_label(self, value):
        max_workers = self.workers_slider.cget("to")
        self.workers_label.configure(text=f"Workers: {int(value)} (Max {int(max_workers)})")

    def start_benchmark(self):
        if self.is_running:
            return
            
        selected_names = [name for name, var in self.circuit_vars.items() if var.get()]
        if not selected_names:
            self.log("⚠ Selecciona al menos un circuito en el panel izquierdo.")
            return

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.terminal_text.configure(state="normal")
        self.terminal_text.delete("1.0", "end")
        self.terminal_text.configure(state="disabled")
        
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.configure(state="disabled")

        # Limpiar frame de gráficos
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        circuits = [bc for bc in DEFAULT_BENCHMARK_CIRCUITS if bc.name in selected_names]
        seeds = list(range(int(self.seeds_slider.get())))
        n_workers = int(self.workers_slider.get())
        backend_name = self.backend_option.get()
        algo_name = self.algo_option.get()
        pop_size = int(self.pop_slider.get())
        gens = int(self.gens_slider.get())
        prob_swap_mutation = float(self.swap_mut_option.get())
        prob_replace_mutation = float(self.replace_mut_option.get())

        self.log(f'Benchmark: {len(circuits)} circuitos × {len(seeds)} semillas = {len(circuits)*len(seeds)} tareas')
        self.log(f'Backend: {backend_name}  |  Algoritmo: {algo_name}  |  Workers: {n_workers}')
        self.log(f'Población: {pop_size}  |  Generaciones: {gens}')
        self.log(f'Mutación categórica: swap={prob_swap_mutation}  |  replace={prob_replace_mutation}')
        self.log('=' * 60)

        threading.Thread(
            target=self._run_benchmark_thread,
            args=(
                circuits,
                seeds,
                n_workers,
                backend_name,
                algo_name,
                pop_size,
                gens,
                prob_swap_mutation,
                prob_replace_mutation,
            ),
            daemon=True,
        ).start()

    def _run_benchmark_thread(
        self,
        circuits,
        seeds,
        n_workers,
        backend_name,
        algo_name,
        pop_size,
        gens,
        prob_swap_mutation,
        prob_replace_mutation,
    ):
        n_total = len(circuits) * len(seeds)
        config = OptimizerConfig(
            algorithm=algo_name,
            population_size=pop_size,
            n_generations=gens,
            objectives=['depth', 'cnot_count'],
            crossover_operator='dpx',
            prob_swap_mutation=prob_swap_mutation,
            prob_replace_mutation=prob_replace_mutation,
            verbose=False,
        )

        backend = get_backend(backend_name)
        result_set = BenchmarkResultSet(backend_name=backend_name, config=config)
        baseline = {bc.name: {} for bc in circuits}

        t0 = time.perf_counter()

        # Paso 1: MO
        self.after(0, self.log, f'\\n[MO] Lanzando {n_total} tareas con {n_workers} workers…')
        self.after(0, lambda: self.progress_label.configure(text="[MO] Optimizando diseños..."))
        self.after(0, self.progress_bar.set, 0)
        
        mo_tasks = [(bc.name, bc.create(), seed, backend, config) for bc in circuits for seed in seeds]
        mo_done = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futs = {executor.submit(_run_mo_one, *t): t for t in mo_tasks}
            for fut in as_completed(futs):
                run = fut.result()
                result_set.runs.append(run)
                mo_done += 1
                tag = '✓' if run.error is None else '✗'
                msg = f'  {tag} [MO] {mo_done}/{n_total}  {run.circuit_name} seed={run.seed}'
                self.after(0, self.log, msg)
                self.after(0, self.progress_bar.set, mo_done / n_total)

        t_mo = time.perf_counter() - t0
        self.after(0, self.log, f'[MO] Completado en {t_mo:.1f} s')

        # Paso 2: Baseline
        self.after(0, self.log, f'\\n[Baseline] Lanzando {n_total} tareas…')
        self.after(0, lambda: self.progress_label.configure(text="[Baseline] Transpilando con Qiskit por defecto..."))
        self.after(0, self.progress_bar.set, 0)

        bl_tasks = [(bc.name, bc.create(), seed, backend, config.optimization_level) for bc in circuits for seed in seeds]
        bl_done = 0

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futs = {executor.submit(_run_baseline_one, *t): t for t in bl_tasks}
            for fut in as_completed(futs):
                bc_name, seed, metrics = fut.result()
                baseline[bc_name][seed] = metrics
                bl_done += 1
                self.after(0, self.log, f'  ✓ [BL] {bl_done}/{n_total}  {bc_name} seed={seed}')
                self.after(0, self.progress_bar.set, bl_done / n_total)

        result_set.total_elapsed_s = time.perf_counter() - t0
        self._last_results = result_set
        self._last_baseline = baseline
        self._last_report = analyze_results(result_set, baseline_results=baseline)

        # Resumen de terminal
        summary_terminal = (
            '\\n' + '=' * 60 + '\\n' +
            f'✓ COMPLETADO: {result_set.n_ok} ok / {result_set.n_failed} fallidas ' +
            f'en {result_set.total_elapsed_s:.1f} s  (workers={n_workers})\\n' +
            '=' * 60 + '\\n\\n' + result_set.summary() + '\\n'
        )
        self.after(0, self.log, summary_terminal)

        # Atualizar tab de resultados
        self.after(0, self._render_results)
        
        # Habilitar UI
        self.after(0, lambda: self.progress_label.configure(text="Completado."))
        self.after(0, self.progress_bar.set, 1)
        self.is_running = False
        self.after(0, lambda: self.run_button.configure(state="normal"))

    def _render_results(self):
        # Escribir informe completo en Resumen
        if self._last_report:
            self.summary_text.configure(state="normal")
            self.summary_text.insert("end", self._last_report.to_text())
            self.summary_text.configure(state="disabled")
            
        # Generar Boxplots
        if self._last_results and self._last_baseline:
            # Limpiar frame de plots previo
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Frame scrollable para poder ver múltiples gráficos si es necesario
            scrollable_plots = ctk.CTkScrollableFrame(self.plot_frame, fg_color="transparent")
            scrollable_plots.pack(fill="both", expand=True, padx=5, pady=5)

            self._render_pareto_fronts(scrollable_plots)
            self._render_boxplots(scrollable_plots)

    def start_tuning(self):
        if self.is_tuning:
            return

        circuit_name = self.t_circuit.get()
        backend_name = self.t_backend.get()
        trials = int(self.t_trials_slider.get())
        seeds = int(self.t_seeds_slider.get())
        workers = int(self.t_workers_slider.get())

        try:
            pmin = int(self.t_pop_min.get())
            pmax = int(self.t_pop_max.get())
            gmin = int(self.t_gen_min.get())
            gmax = int(self.t_gen_max.get())
        except ValueError:
            self.t_log("⚠ Error: Los rangos de población y generaciones deben ser enteros.")
            return

        try:
            swap_choices = tuple(
                sorted({float(x.strip()) for x in self.t_swap_choices.get().split(',') if x.strip()})
            )
            replace_choices = tuple(
                sorted({float(x.strip()) for x in self.t_replace_choices.get().split(',') if x.strip()})
            )
        except ValueError:
            self.t_log("⚠ Error: Las categorías de mutación deben ser floats separados por comas.")
            return

        if not swap_choices or not replace_choices:
            self.t_log("⚠ Error: Debes indicar al menos una categoría para cada mutación.")
            return

        objs = []
        if self.t_obj_depth.get(): objs.append("depth")
        if self.t_obj_cnot.get(): objs.append("cnot_count")

        algs = []
        if self.t_alg_nsga2.get(): algs.append("nsga2")
        if self.t_alg_moead.get(): algs.append("moead")

        if not objs or not algs:
            self.t_log("⚠ Selecciona al menos un objetivo y un algoritmo.")
            return

        self.is_tuning = True
        self.t_run_button.configure(state="disabled")
        self.t_best_config_btn.configure(state="disabled")
        self.t_terminal_text.configure(state="normal")
        self.t_terminal_text.delete("1.0", "end")
        self.t_terminal_text.configure(state="disabled")

        for widget in self.t_plot_frame.winfo_children():
            widget.destroy()

        self.t_log(f'Tuning: {trials} trials × {seeds} seeds')
        self.t_log(f'Workers: {workers}')
        self.t_log(f'Circuito: {circuit_name} | Backend: {backend_name}')
        self.t_log(f'Pop Rango: [{pmin}, {pmax}] | Gen Rango: [{gmin}, {gmax}]')
        self.t_log(f'Categorías swap: {list(swap_choices)} | Categorías replace: {list(replace_choices)}')
        self.t_log(f'Objetivos: {objs} | Algoritmos: {algs}')
        self.t_log('=' * 60)

        threading.Thread(
            target=self._run_tuning_thread,
            args=(
                circuit_name,
                backend_name,
                trials,
                seeds,
                workers,
                pmin,
                pmax,
                gmin,
                gmax,
                swap_choices,
                replace_choices,
                objs,
                algs,
            ),
            daemon=True,
        ).start()

    def _run_tuning_thread(
        self,
        c_name,
        b_name,
        t,
        s,
        w,
        pmin,
        pmax,
        gmin,
        gmax,
        swap_choices,
        replace_choices,
        objs,
        algs,
    ):
        self.after(0, lambda: self.t_progress_label.configure(text="Ejecutando Optuna (ver salida en consola del script original)..."))
        t0 = time.perf_counter()

        try:
            bc = next(bc for bc in DEFAULT_BENCHMARK_CIRCUITS if bc.name == c_name)
            circuit = bc.create()
            backend = get_backend(b_name)

            space = HyperparameterSpace(
                population_size_range=(pmin, pmax),
                n_generations_range=(gmin, gmax),
                prob_swap_mutation_choices=swap_choices,
                prob_replace_mutation_choices=replace_choices,
                algorithms=algs,
            )

            self._tuner = LayoutTuner(
                circuit=circuit,
                backend=backend,
                n_trials=t,
                n_seeds=s,
                n_jobs=w,
                space=space,
                objectives=objs,
                study_name=f'tuning_{c_name}_{b_name}',
            )

            # Para capturar logs, redirigimos stdout temporalmente
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            self._tuner.tune(show_progress_bar=False) # Progreso en GUI es complejo con optuna bar
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Print logs if available
            if output:
                 self.after(0, self.t_log, output)

            t_t = time.perf_counter() - t0
            self.after(0, self.t_log, f'\\n✓ Tuning completado en {t_t:.1f} s\\n')
            
            summary = self._tuner.summary()
            self.after(0, self.t_log, summary)
            
            best = self._tuner.best_config()
            self.after(0, self.t_log, f'\\nMejor configuración:\\n{best}')

            # Plots
            self.after(0, self._render_tuning_plots)

        except Exception as e:
            self.after(0, self.t_log, f'\\n⚠ Error durante el tuning: {str(e)}')

        finally:
            self.after(0, lambda: self.t_progress_label.configure(text="Completado."))
            self.after(0, lambda: self.t_run_button.configure(state="normal"))
            if self._tuner and getattr(self._tuner, "_best_config", None):
                self.after(0, lambda: self.t_best_config_btn.configure(state="normal"))
            self.is_tuning = False

    def _render_tuning_plots(self):
        try:
            import optuna.visualization as vis
            study = self._tuner.study
            
            # optuna plots output a plotly Figure. We need to save it to an image, then put in matplotlib
            import plotly.io as pio
            
            # This requires kaleido or orca. Let's do a simple workaround without saving images if possible,
            # Or use matplotlib directly for study visualization
            from optuna.visualization.matplotlib import plot_optimization_history
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            fig, ax = plt.subplots(figsize=(6, 4))
            plot_optimization_history(study, ax=ax)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.t_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        except Exception as e:
            self.t_log(f"No se pudieron generar las gráficas: {str(e)}\n(Puede que falte plotly/kaleido para exports, o sea versión de matplotlib antigua)")

    def _copy_best_to_benchmark(self):
        try:
            best = self._tuner.best_config()
            self.pop_slider.set(best.population_size)
            self.gens_slider.set(best.n_generations)
            self._update_pop_label(best.population_size)
            self._update_gens_label(best.n_generations)
            self.swap_mut_option.set(str(best.prob_swap_mutation))
            self.replace_mut_option.set(str(best.prob_replace_mutation))
            self._update_swap_mut_label(str(best.prob_swap_mutation))
            self._update_replace_mut_label(str(best.prob_replace_mutation))
            
            if best.algorithm in ['nsga2', 'moead']:
                 self.algo_option.set(best.algorithm)
            
            self.master_tabview.set("Modo Benchmark")
            self.t_log("✓ Configuración transferida al panel de Benchmark")
        except Exception as e:
            self.t_log(f"Error copiando config: {str(e)}")

    def _render_pareto_fronts(self, parent_frame):
        circuit_names = self._last_results.circuit_names
        runs_first = self._last_results.runs_for_circuit(circuit_names[0])
        if not runs_first: return
        obj_names = runs_first[0].result.objective_names
        
        if len(obj_names) < 2:
            return

        fig, axes = plt.subplots(1, len(circuit_names), figsize=(5 * len(circuit_names), 4), squeeze=False)
        fig.suptitle('Frente de Pareto (MO) vs Baseline Qiskit', fontsize=12, fontweight='bold')

        for ci, cname in enumerate(circuit_names):
            ax = axes[0][ci]

            runs = self._last_results.runs_for_circuit(cname)
            for run in runs:
                if run.result.pareto_fitness is not None and len(run.result.pareto_fitness) > 0 and run.result.pareto_fitness.shape[1] >= 2:
                    F = run.result.pareto_fitness
                    # Ordenar por el primer objetivo para conectar los puntos con una línea si se desea (opcional)
                    idx = np.argsort(F[:, 0])
                    F_sorted = F[idx]
                    
                    # Dibujar todos los puntos del frente para esta run
                    ax.scatter(F_sorted[:, 0], F_sorted[:, 1], alpha=0.5, s=20, label='MO' if run is runs[0] else None)
                    # Opcional: unir los puntos del frente con una línea tenue
                    ax.plot(F_sorted[:, 0], F_sorted[:, 1], alpha=0.3, linewidth=1)

            bl = self._last_baseline.get(cname, {}) if self._last_baseline else {}
            if bl:
                bl_x = [v[obj_names[0]] for v in bl.values() if obj_names[0] in v and v[obj_names[0]] != float('inf')]
                bl_y = [v[obj_names[1]] for v in bl.values() if obj_names[1] in v and v[obj_names[1]] != float('inf')]
                if bl_x and bl_y:
                    ax.scatter(bl_x, bl_y, marker='X', s=80, color='#e74c3c', edgecolors='k', linewidths=0.5, zorder=5, label='Qiskit default')

            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1] if len(obj_names) > 1 else '')
            ax.set_title(cname)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=False, pady=(0, 20))

    def _render_boxplots(self, parent_frame):
        circuit_names = self._last_results.circuit_names
        n_circuits = len(circuit_names)
        obj_names = self._last_results.runs_for_circuit(circuit_names[0])[0].result.objective_names
        n_obj = len(obj_names)

        fig, axes = plt.subplots(
            n_obj, n_circuits,
            figsize=(5 * n_circuits, 3.5 * n_obj),
            squeeze=False,
        )
        fig.suptitle('MO vs Qiskit por defecto — Distribución por semilla', fontsize=12, fontweight='bold')

        for ci, cname in enumerate(circuit_names):
            for oi, oname in enumerate(obj_names):
                ax = axes[oi][ci]

                # Valores MO
                mo_values = self._last_results.best_per_seed(cname, oi)

                # Valores Baseline
                bl = self._last_baseline.get(cname, {}) if self._last_baseline else {}
                bl_values = [v[oname] for v in bl.values() if oname in v and v[oname] != float('inf')]

                data_to_plot = [mo_values, bl_values] if bl_values else [mo_values]
                labels = ['MO', 'Qiskit'] if bl_values else ['MO']
                colors = ['#aec6e8', '#f4a582']

                bp = ax.boxplot(data_to_plot, vert=True, patch_artist=True,
                                tick_labels=labels,
                                medianprops=dict(color='#c0392b', linewidth=2))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

                # Jitter
                rng = np.random.default_rng(0)
                for pos_idx, vals in enumerate(data_to_plot):
                    jitter = rng.uniform(-0.1, 0.1, len(vals))
                    ax.scatter([pos_idx + 1 + j for j in jitter], vals,
                               alpha=0.6, s=18, color='#2c3e50', zorder=3)

                ax.set_title(f'{cname}\n{oname}', fontsize=9)
                ax.set_ylabel(oname if ci == 0 else '')

                # Anotación porcentaje
                if bl_values and mo_values:
                    mo_med = np.median(mo_values)
                    bl_med = np.median(bl_values)
                    if bl_med > 0:
                        pct = (bl_med - mo_med) / bl_med * 100
                        sign = '+' if pct >= 0 else ''
                        ax.annotate(f'{sign}{pct:.1f}%', xy=(0.5, 0.02),
                                    xycoords='axes fraction', ha='center',
                                    fontsize=8, color='green' if pct > 0 else 'red',
                                    fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = BenchmarkGUI()
    app.mainloop()
