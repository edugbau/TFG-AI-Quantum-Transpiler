"""
GUI para pruebas aisladas del Módulo de Aprendizaje por Refuerzo.

Interfaz gráfica con CustomTkinter que permite:
  - Configurar circuito, coupling map, modo, algoritmo e hiperparámetros.
  - Entrenar un agente PPO/DQN con visualización de progreso en tiempo real.
  - Evaluar el agente entrenado ejecutando un episodio completo paso a paso.
  - Visualizar gráficos de recompensa del entrenamiento.

Ejecución:
    python -m src.rl_module.gui.rl_gui
"""

import sys
import pathlib
import threading
import time
import logging
import os
import importlib
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk

# Ensure src is in the python path if executed standalone
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

from qiskit import QuantumCircuit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from src.rl_module.environment import QuantumTranspilationEnv
from src.rl_module.agent import QuantumRLAgent
from src.rl_module.model_metadata import build_run_metadata, save_run_metadata
from src.rl_module.routing_mask import (
    DEFAULT_NEW_MASK_SEMANTICS,
    FRONTIER_RESTRICTED_EDGES_V3,
    FRONTIER_RESTRICTED_EDGES_V4,
    resolve_routing_mask_config,
)
from src.rl_module.training import set_global_seeds
from src.rl_module.gui.routing_view import RoutingView
from src.rl_module.gui.synthesis_view import SynthesisView
from src.rl_module.gui.evaluation_panel import (
    EpisodeInspectorPanel,
    EvaluationStepRecord,
    frontier_entry_to_dict,
)

logger = logging.getLogger(__name__)


MASKED_ROUTING_SEMANTICS = DEFAULT_NEW_MASK_SEMANTICS
ROUTING_ALGORITHMS = ["PPO", "DQN", "MaskablePPO"]
SYNTHESIS_ALGORITHMS = ["PPO", "DQN"]

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


def _make_run_dir(base_dir: str, prefix: str = "run") -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(base_dir, f"{prefix}_{run_id}")


def _normalize_masked_routing_config(cfg: dict) -> dict:
    normalized_cfg = dict(cfg)
    is_masked_routing = (
        normalized_cfg.get("mode") == "routing"
        and normalized_cfg.get("algorithm") == "MaskablePPO"
    )
    normalized_cfg["masked"] = is_masked_routing
    if is_masked_routing:
        normalized_cfg["mask_semantics"] = normalized_cfg.get(
            "mask_semantics",
            MASKED_ROUTING_SEMANTICS,
        )
        if normalized_cfg["mask_semantics"] in {FRONTIER_RESTRICTED_EDGES_V3, FRONTIER_RESTRICTED_EDGES_V4}:
            normalized_cfg["routing_mask_config"] = resolve_routing_mask_config(
                normalized_cfg.get("routing_mask_config"),
                num_qubits=normalized_cfg["circuit"].num_qubits,
            )
    else:
        normalized_cfg.pop("mask_semantics", None)
        normalized_cfg.pop("routing_mask_config", None)
    return normalized_cfg


def _build_eval_callback(eval_env, cfg: dict, run_model_dir: str, run_log_dir: str, eval_freq: int):
    if not (cfg.get("mode") == "routing" and cfg.get("algorithm") == "MaskablePPO"):
        return EvalCallback(
            eval_env,
            best_model_save_path=run_model_dir,
            log_path=run_log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

    try:
        sb3_contrib_callbacks = importlib.import_module("sb3_contrib.common.maskable.callbacks")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MaskablePPO requiere instalar sb3-contrib para la evaluacion GUI."
        ) from exc

    return sb3_contrib_callbacks.MaskableEvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )


# ===========================================================================
#  Circuitos y Coupling Maps predefinidos
# ===========================================================================

PRESET_CIRCUITS = {
    "GHZ-3": lambda: _make_ghz(3),
    "GHZ-5": lambda: _make_ghz(5),
    "Linear CX-4": lambda: _make_linear_cx(4),
    "Random CX-5": lambda: _make_random_cx(5, seed=42),
    "Bell State": lambda: _make_bell(),
    "CX Chain-3": lambda: _make_cx_chain(3),
}

PRESET_COUPLINGS = {
    "Lineal-3  (0-1-2)": [(0, 1), (1, 2)],
    "Lineal-4  (0-1-2-3)": [(0, 1), (1, 2), (2, 3)],
    "Lineal-5  (0-1-2-3-4)": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "Anillo-4  (0-1-2-3-0)": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "Anillo-5  (0-1-2-3-4-0)": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    "Estrella-4 (0→1,2,3)": [(0, 1), (0, 2), (0, 3)],
    "T-shape-5 (H+V)": [(0, 1), (1, 2), (1, 3), (3, 4)],
    "Grid-4 (2x2)": [(0, 1), (1, 3), (0, 2), (2, 3)],
}

SYNTHESIS_BASIS_PROFILES = {
    "Clifford-CZ": ["cz", "rz", "sx", "x"],
    "Clifford-ECR": ["ecr", "rz", "sx", "x"],
}


def _make_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"GHZ-{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def _make_linear_cx(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"LinearCX-{n}")
    for i in range(n - 1):
        qc.cx(i, i + 1)
    # Añadir una CX no adyacente para forzar SWAPs
    if n >= 4:
        qc.cx(0, n - 1)
    return qc


def _make_random_cx(n: int, seed: int = 42) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, name=f"RandomCX-{n}")
    qc.h(0)
    for _ in range(n * 2):
        q1, q2 = rng.choice(n, size=2, replace=False)
        qc.cx(int(q1), int(q2))
    return qc


def _make_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, name="Bell")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _make_cx_chain(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"CXChain-{n}")
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


# ===========================================================================
#  Callback para capturar métricas durante el entrenamiento
# ===========================================================================

class GUIProgressCallback(BaseCallback):
    """Callback de SB3 que captura episodios y notifica a la GUI."""

    def __init__(self, gui: "RLBenchmarkGUI", total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.gui = gui
        self.total_timesteps = total_timesteps
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_ep_reward = 0.0
        self._current_ep_len = 0

    def _on_step(self) -> bool:
        # Actualizar progreso
        progress = self.num_timesteps / self.total_timesteps
        self.gui.after(0, self.gui._progress_bar.set, min(progress, 1.0))
        self.gui.after(
            0, self.gui._progress_label.configure,
            {"text": f"Entrenando... {self.num_timesteps}/{self.total_timesteps} steps"}
        )

        # Capturar rewards de episodios del Monitor
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
                ep_num = len(self.episode_rewards)
                msg = f"  Ep {ep_num}: reward={ep_info['r']:.1f}  len={ep_info['l']}"
                self.gui.after(0, self.gui._log, msg)

        return True


# ===========================================================================
#  Aplicación principal
# ===========================================================================

class RLBenchmarkGUI(ctk.CTk):
    """Interfaz gráfica de pruebas para el módulo RL."""

    def __init__(self):
        super().__init__()

        self.title("RL Module — Entrenamiento y Evaluación")
        self.geometry("1300x850")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Estado
        self._is_training = False
        self._agent: QuantumRLAgent | None = None
        self._env: QuantumTranspilationEnv | None = None
        self._last_callback: GUIProgressCallback | None = None
        self._training_cfg: dict | None = None  # Configuración usada en el entrenamiento
        self._eval_log: list[EvaluationStepRecord] = []

        self._create_sidebar()
        self._create_main_frame()

    # -----------------------------------------------------------------------
    #  Sidebar
    # -----------------------------------------------------------------------
    def _create_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=310, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(20, weight=1)

        ctk.CTkLabel(
            sidebar, text="⚙ Configuración RL",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 15))

        row = 1

        # --- Circuito ---
        ctk.CTkLabel(sidebar, text="Circuito:", anchor="w").grid(
            row=row, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        row += 1
        self._circuit_option = ctk.CTkOptionMenu(
            sidebar, values=list(PRESET_CIRCUITS.keys())
        )
        self._circuit_option.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        # --- Coupling Map ---
        ctk.CTkLabel(sidebar, text="Coupling Map:", anchor="w").grid(
            row=row, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        row += 1
        self._coupling_option = ctk.CTkOptionMenu(
            sidebar, values=list(PRESET_COUPLINGS.keys())
        )
        self._coupling_option.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        # --- Modo ---
        ctk.CTkLabel(sidebar, text="Modo:", anchor="w").grid(
            row=row, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        row += 1
        self._mode_option = ctk.CTkOptionMenu(
            sidebar,
            values=["routing", "synthesis"],
            command=self._on_mode_changed,
        )
        self._mode_option.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        self._mode_view_container = ctk.CTkFrame(sidebar, fg_color="transparent")
        self._mode_view_container.grid(row=row, column=0, padx=20, pady=0, sticky="ew")
        self._mode_view_container.grid_columnconfigure(0, weight=1)

        self._routing_view = RoutingView(
            self._mode_view_container,
            on_lookahead_change=self._update_lookahead_label,
        )
        self._routing_view.grid(row=0, column=0, sticky="ew")
        self._synthesis_view = SynthesisView(
            self._mode_view_container,
            basis_profiles=SYNTHESIS_BASIS_PROFILES.keys(),
        )
        self._synthesis_view.grid(row=0, column=0, sticky="ew")

        self._frontier_option = self._routing_view.frontier_option
        self._lookahead_label = self._routing_view.lookahead_label
        self._lookahead_slider = self._routing_view.lookahead_slider
        self._basis_profile_option = self._synthesis_view.basis_profile_option
        self._lookahead_slider.set(10)
        row += 1

        # --- Algoritmo RL ---
        ctk.CTkLabel(sidebar, text="Algoritmo RL:", anchor="w").grid(
            row=row, column=0, padx=20, pady=(10, 0), sticky="w"
        )
        row += 1
        self._algo_option = ctk.CTkOptionMenu(sidebar, values=ROUTING_ALGORITHMS)
        self._algo_option.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        # --- Timesteps ---
        self._timesteps_label = ctk.CTkLabel(sidebar, text="Timesteps: 5,000", anchor="w")
        self._timesteps_label.grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        row += 1
        self._timesteps_slider = ctk.CTkSlider(
            sidebar, from_=100, to=50000, number_of_steps=499,
            command=lambda v: self._timesteps_label.configure(
                text=f"Timesteps: {int(v):,}"
            ),
        )
        self._timesteps_slider.set(5000)
        self._timesteps_slider.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        # --- Max Steps / Episode ---
        self._maxsteps_label = ctk.CTkLabel(sidebar, text="Max Steps/Ep: 200", anchor="w")
        self._maxsteps_label.grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        row += 1
        self._maxsteps_slider = ctk.CTkSlider(
            sidebar, from_=20, to=2000, number_of_steps=198,
            command=lambda v: self._maxsteps_label.configure(
                text=f"Max Steps/Ep: {int(v)}"
            ),
        )
        self._maxsteps_slider.set(200)
        self._maxsteps_slider.grid(row=row, column=0, padx=20, pady=(0, 10), sticky="ew")
        row += 1

        # --- Seed ---
        self._seed_label = ctk.CTkLabel(sidebar, text="Seed: 42", anchor="w")
        self._seed_label.grid(row=row, column=0, padx=20, pady=(10, 0), sticky="w")
        row += 1
        self._seed_slider = ctk.CTkSlider(
            sidebar, from_=0, to=100, number_of_steps=100,
            command=lambda v: self._seed_label.configure(
                text=f"Seed: {int(v)}"
            ),
        )
        self._seed_slider.set(42)
        self._seed_slider.grid(row=row, column=0, padx=20, pady=(0, 15), sticky="ew")
        row += 1

        # --- Botones ---
        self._train_button = ctk.CTkButton(
            sidebar, text="▶  Entrenar Agente",
            command=self._start_training,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
        )
        self._train_button.grid(row=row, column=0, padx=20, pady=(5, 5), sticky="ew")
        row += 1

        self._eval_button = ctk.CTkButton(
            sidebar, text="🔍  Evaluar Episodio",
            command=self._start_evaluation,
            state="disabled",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="#27ae60",
            hover_color="#1e8449",
        )
        self._eval_button.grid(row=row, column=0, padx=20, pady=(5, 20), sticky="ew")
        row += 1

        self._on_mode_changed(self._mode_option.get())

    def _update_lookahead_label(self, value):
        self._lookahead_label.configure(text=f"Lookahead: {int(value)}")

    def _on_mode_changed(self, mode: str):
        if mode == "synthesis":
            self._routing_view.hide()
            self._synthesis_view.show()
            supported_algorithms = SYNTHESIS_ALGORITHMS
            self._algo_option.configure(values=supported_algorithms)
            if self._algo_option.get() not in supported_algorithms:
                self._algo_option.set(supported_algorithms[0])
            return

        self._synthesis_view.hide()
        self._routing_view.show()
        self._algo_option.configure(values=ROUTING_ALGORITHMS)

    # -----------------------------------------------------------------------
    #  Main Frame
    # -----------------------------------------------------------------------
    def _create_main_frame(self):
        main = ctk.CTkFrame(self)
        main.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Tab View
        self._tabview = ctk.CTkTabview(main)
        self._tabview.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        # --- Tab: Progreso ---
        tab_progress = self._tabview.add("Progreso & Terminal")
        tab_progress.grid_rowconfigure(2, weight=1)
        tab_progress.grid_columnconfigure(0, weight=1)

        self._progress_label = ctk.CTkLabel(
            tab_progress, text="Esperando inicio...",
            font=ctk.CTkFont(weight="bold"),
        )
        self._progress_label.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        self._progress_bar = ctk.CTkProgressBar(tab_progress)
        self._progress_bar.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        self._progress_bar.set(0)

        self._terminal = ctk.CTkTextbox(
            tab_progress, state="disabled",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self._terminal.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        # --- Tab: Evaluación ---
        tab_eval = self._tabview.add("Evaluación Episodio")
        tab_eval.grid_rowconfigure(0, weight=1)
        tab_eval.grid_columnconfigure(0, weight=1)
        tab_eval.grid_columnconfigure(1, weight=2)

        self._eval_inspector = EpisodeInspectorPanel(tab_eval)
        self._eval_inspector.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)

        self._eval_terminal = ctk.CTkTextbox(
            tab_eval, state="disabled",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self._eval_terminal.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)

        # --- Tab: Gráficos ---
        tab_plots = self._tabview.add("Gráficos")
        tab_plots.grid_rowconfigure(0, weight=1)
        tab_plots.grid_columnconfigure(0, weight=1)

        self._plot_frame = ctk.CTkFrame(tab_plots, fg_color="transparent")
        self._plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # -----------------------------------------------------------------------
    #  Log helpers
    # -----------------------------------------------------------------------
    def _log(self, text: str):
        self._terminal.configure(state="normal")
        self._terminal.insert("end", text + "\n")
        self._terminal.see("end")
        self._terminal.configure(state="disabled")

    def _eval_log_write(self, text: str):
        self._eval_terminal.configure(state="normal")
        self._eval_terminal.insert("end", text + "\n")
        self._eval_terminal.see("end")
        self._eval_terminal.configure(state="disabled")

    def _clear_terminal(self):
        self._terminal.configure(state="normal")
        self._terminal.delete("1.0", "end")
        self._terminal.configure(state="disabled")

    def _clear_eval_terminal(self):
        self._eval_terminal.configure(state="normal")
        self._eval_terminal.delete("1.0", "end")
        self._eval_terminal.configure(state="disabled")

    def _append_eval_record(self, record: EvaluationStepRecord):
        if not hasattr(self, "_eval_log"):
            self._eval_log = []

        self._eval_log.append(record)

        inspector = getattr(self, "_eval_inspector", None)
        if inspector is not None and hasattr(inspector, "set_records"):
            inspector.set_records(self._eval_log)

    # -----------------------------------------------------------------------
    #  Leer configuración del sidebar
    # -----------------------------------------------------------------------
    def _get_config(self) -> dict:
        circuit_name = self._circuit_option.get()
        coupling_name = self._coupling_option.get()
        return {
            "circuit_name": circuit_name,
            "circuit": PRESET_CIRCUITS[circuit_name](),
            "coupling_map": PRESET_COUPLINGS[coupling_name],
            "coupling_name": coupling_name,
            "mode": self._mode_option.get(),
            "basis_gates": SYNTHESIS_BASIS_PROFILES[self._basis_profile_option.get()],
            "frontier_mode": self._frontier_option.get(),
            "algorithm": self._algo_option.get(),
            "timesteps": int(self._timesteps_slider.get()),
            "max_steps": int(self._maxsteps_slider.get()),
            "lookahead": int(self._lookahead_slider.get()),
            "seed": int(self._seed_slider.get()),
        }

    # -----------------------------------------------------------------------
    #  Entrenamiento
    # -----------------------------------------------------------------------
    def _start_training(self):
        if self._is_training:
            return

        cfg = _normalize_masked_routing_config(self._get_config())

        if cfg["mode"] == "synthesis" and cfg["algorithm"] == "MaskablePPO":
            self._log(
                "⚠ MaskablePPO solo esta disponible para routing en la GUI. "
                "Selecciona PPO o DQN para synthesis."
            )
            return

        # Validar compatibilidad circuito ↔ coupling map
        cm_qubits = set()
        for a, b in cfg["coupling_map"]:
            cm_qubits.add(a)
            cm_qubits.add(b)
        if cfg["circuit"].num_qubits > len(cm_qubits):
            self._log(
                f"⚠ El circuito tiene {cfg['circuit'].num_qubits} qubits pero el "
                f"coupling map solo tiene {len(cm_qubits)} qubits. "
                "Selecciona un coupling map compatible."
            )
            return

        self._is_training = True
        self._train_button.configure(state="disabled")
        self._eval_button.configure(state="disabled")
        self._clear_terminal()
        self._progress_bar.set(0)

        self._log(f"Circuito: {cfg['circuit_name']}  |  Qubits: {cfg['circuit'].num_qubits}")
        self._log(f"Coupling: {cfg['coupling_name']}")
        self._log(f"Modo: {cfg['mode']}  |  Frontier: {cfg['frontier_mode']}  |  Algoritmo: {cfg['algorithm']}")
        if cfg.get("routing_mask_config") is not None:
            mask_cfg = cfg["routing_mask_config"]
            self._log(
                f"Mascara {cfg['mask_semantics'].rsplit('.', 1)[-1]}: "
                f"cycle_window={mask_cfg.cycle_window}  |  "
                f"stagnation_patience={mask_cfg.stagnation_patience}  |  "
                f"sabre_top_k={mask_cfg.sabre_top_k}"
            )
        if cfg["mode"] == "synthesis":
            self._log(f"Basis Gates: {cfg['basis_gates']}")
        self._log(f"Timesteps: {cfg['timesteps']:,}  |  Max Steps/Ep: {cfg['max_steps']}  |  Lookahead: {cfg['lookahead']}")
        self._log(f"Seed: {cfg['seed']}")
        self._log("=" * 60)

        self._training_cfg = cfg  # Guardar para reutilizar en evaluación

        threading.Thread(
            target=self._training_thread, args=(cfg,), daemon=True
        ).start()

    def _training_thread(self, cfg: dict):
        t0 = time.perf_counter()
        training_succeeded = False

        try:
            set_global_seeds(cfg["seed"])

            # Crear entorno
            raw_env = QuantumTranspilationEnv(
                target_circuit=cfg["circuit"],
                coupling_map=cfg["coupling_map"],
                mode=cfg["mode"],
                frontier_mode=cfg["frontier_mode"],
                lookahead_window=cfg["lookahead"],
                max_steps=cfg["max_steps"],
                basis_gates=cfg.get("basis_gates") if cfg["mode"] == "synthesis" else None,
                mask_semantics=cfg.get("mask_semantics"),
                routing_mask_config=cfg.get("routing_mask_config"),
            )
            raw_env.reset(seed=cfg["seed"])
            self._env = Monitor(raw_env)

            eval_raw_env = QuantumTranspilationEnv(
                target_circuit=cfg["circuit"],
                coupling_map=cfg["coupling_map"],
                mode=cfg["mode"],
                frontier_mode=cfg["frontier_mode"],
                lookahead_window=cfg["lookahead"],
                max_steps=cfg["max_steps"],
                basis_gates=cfg.get("basis_gates") if cfg["mode"] == "synthesis" else None,
                mask_semantics=cfg.get("mask_semantics"),
                routing_mask_config=cfg.get("routing_mask_config"),
            )
            eval_raw_env.reset(seed=cfg["seed"])
            eval_env = Monitor(eval_raw_env)

            self.after(0, self._log, "[1/3] Entorno creado ✓")

            # Crear agente
            self._agent = QuantumRLAgent(
                env=self._env,
                algorithm=cfg["algorithm"],
                verbose=0,
                seed=cfg["seed"],
            )
            self.after(0, self._log, f"[2/3] Agente {cfg['algorithm']} creado ✓")
            self.after(0, self._log, f"      Dispositivo: {self._agent.device.upper()}")
            self.after(0, self._log, "")
            self.after(0, self._log, "Entrenando...")

            # Callback de progreso
            callback = GUIProgressCallback(
                gui=self, total_timesteps=cfg["timesteps"]
            )
            run_model_dir = _make_run_dir("./experiments/models/rl_models", prefix="gui_rl")
            run_log_dir = _make_run_dir("./experiments/logs/rl_logs", prefix="gui_rl")
            os.makedirs(run_model_dir, exist_ok=True)
            os.makedirs(run_log_dir, exist_ok=True)
            save_run_metadata(
                run_model_dir,
                build_run_metadata(
                    mode=cfg["mode"],
                    algorithm=cfg["algorithm"],
                    seed=cfg["seed"],
                    frontier_mode=cfg["frontier_mode"],
                    lookahead_window=cfg["lookahead"],
                    max_steps=cfg["max_steps"],
                    basis_gates=(
                        cfg.get("basis_gates") if cfg["mode"] == "synthesis" else None
                    ),
                    mask_semantics=cfg.get("mask_semantics"),
                    routing_mask_config=cfg.get("routing_mask_config"),
                ),
            )

            callbacks = [callback]
            should_track_best = cfg["timesteps"] > 1
            if should_track_best:
                eval_callback = _build_eval_callback(
                    eval_env,
                    cfg,
                    run_model_dir,
                    run_log_dir,
                    max(1, cfg["timesteps"] // 2),
                )
                callbacks.append(eval_callback)

            # Entrenar
            self._agent.train(
                total_timesteps=cfg["timesteps"],
                callbacks=callbacks,
                progress_bar=False,
            )
            self._last_callback = callback

            final_model_path = os.path.join(run_model_dir, "final_model.zip")
            self._agent.save(final_model_path)
            cfg["last_model_path"] = final_model_path
            cfg["run_model_dir"] = run_model_dir
            cfg["run_log_dir"] = run_log_dir

            best_model_path = os.path.join(run_model_dir, "best_model.zip")
            if should_track_best and os.path.exists(best_model_path):
                cfg["best_model_path"] = best_model_path
            else:
                cfg["best_model_path"] = None

            elapsed = time.perf_counter() - t0
            n_eps = len(callback.episode_rewards)

            self.after(0, self._log, "")
            self.after(0, self._log, "=" * 60)
            self.after(0, self._log, f"[3/3] Entrenamiento completado ✓")
            self.after(0, self._log, f"      Tiempo: {elapsed:.1f}s  |  Episodios: {n_eps}")

            if n_eps > 0:
                avg_r = np.mean(callback.episode_rewards[-min(10, n_eps):])
                self.after(0, self._log, f"      Reward media (últimos 10 ep): {avg_r:.1f}")

            # Renderizar gráficos
            self.after(0, self._render_training_plots)
            self.after(0, lambda: self._progress_label.configure(text="Entrenamiento completado."))
            self.after(0, self._progress_bar.set, 1.0)
            training_succeeded = True

        except Exception as e:
            self.after(0, self._log, f"\n⚠ Error durante entrenamiento: {e}")
            import traceback
            self.after(0, self._log, traceback.format_exc())

        finally:
            self._is_training = False
            self.after(0, lambda: self._train_button.configure(state="normal"))
            if training_succeeded:
                self.after(0, lambda: self._eval_button.configure(state="normal"))

    # -----------------------------------------------------------------------
    #  Gráficos de entrenamiento
    # -----------------------------------------------------------------------
    def _render_training_plots(self):
        # Limpiar plots anteriores
        for w in self._plot_frame.winfo_children():
            w.destroy()

        cb = self._last_callback
        if cb is None or len(cb.episode_rewards) == 0:
            return

        rewards = cb.episode_rewards
        lengths = cb.episode_lengths

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        try:
            fig.suptitle("Métricas de Entrenamiento", fontsize=13, fontweight="bold")

            # --- Reward por episodio ---
            ax1 = axes[0]
            ax1.plot(rewards, alpha=0.4, linewidth=1, color="#3498db", label="Reward")
            # Media móvil
            if len(rewards) >= 5:
                window = min(20, len(rewards) // 2)
                moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax1.plot(
                    range(window - 1, len(rewards)),
                    moving_avg,
                    linewidth=2, color="#e74c3c", label=f"Media móvil ({window})",
                )
            ax1.set_xlabel("Episodio")
            ax1.set_ylabel("Reward Total")
            ax1.set_title("Reward por Episodio")
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # --- Longitud de episodio ---
            ax2 = axes[1]
            ax2.plot(lengths, alpha=0.4, linewidth=1, color="#2ecc71", label="Longitud")
            if len(lengths) >= 5:
                window = min(20, len(lengths) // 2)
                moving_avg_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
                ax2.plot(
                    range(window - 1, len(lengths)),
                    moving_avg_l,
                    linewidth=2, color="#8e44ad", label=f"Media móvil ({window})",
                )
            ax2.set_xlabel("Episodio")
            ax2.set_ylabel("Steps")
            ax2.set_title("Longitud de Episodio")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0, 1, 0.93])

            canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        finally:
            # Cerrar la figura de matplotlib para liberar memoria
            plt.close(fig)

    # -----------------------------------------------------------------------
    #  Evaluación
    # -----------------------------------------------------------------------
    def _start_evaluation(self):
        if self._is_training or self._agent is None:
            return

        self._eval_button.configure(state="disabled")
        self._eval_log = []
        self._eval_inspector.set_records(self._eval_log)
        self._clear_eval_terminal()
        self._tabview.set("Evaluación Episodio")

        threading.Thread(target=self._evaluation_thread, daemon=True).start()

    def _evaluation_thread(self):
        try:
            # Reutilizar la configuración del entrenamiento para que el entorno
            # de evaluación sea idéntico al que se usó para entrenar.
            # Si no hay entrenamiento previo, leer la config actual del sidebar.
            cfg_source = self._training_cfg if self._training_cfg is not None else self._get_config()
            cfg = _normalize_masked_routing_config(cfg_source)
            set_global_seeds(cfg["seed"])

            # Crear entorno limpio para evaluación
            eval_env = QuantumTranspilationEnv(
                target_circuit=cfg["circuit"],
                coupling_map=cfg["coupling_map"],
                mode=cfg["mode"],
                frontier_mode=cfg["frontier_mode"],
                lookahead_window=cfg["lookahead"],
                max_steps=cfg["max_steps"],
                basis_gates=cfg.get("basis_gates") if cfg["mode"] == "synthesis" else None,
                mask_semantics=cfg.get("mask_semantics"),
                routing_mask_config=cfg.get("routing_mask_config"),
            )

            eval_agent = self._agent
            best_model_path = cfg.get("best_model_path")
            last_model_path = cfg.get("last_model_path")
            run_model_dir = cfg.get("run_model_dir")
            if (
                best_model_path
                and run_model_dir
                and os.path.dirname(best_model_path) == run_model_dir
                and os.path.exists(best_model_path)
            ):
                eval_agent = QuantumRLAgent.load(
                    best_model_path,
                    env=eval_env,
                    algorithm=cfg["algorithm"],
                )
            elif last_model_path and run_model_dir and os.path.dirname(last_model_path) == run_model_dir and os.path.exists(last_model_path):
                eval_agent = QuantumRLAgent.load(
                    last_model_path,
                    env=eval_env,
                    algorithm=cfg["algorithm"],
                )

            obs, info = eval_env.reset(seed=cfg["seed"])

            self.after(0, self._eval_log_write, "=" * 70)
            self.after(0, self._eval_log_write, "  EVALUACIÓN DE EPISODIO (Política Determinista)")
            self.after(0, self._eval_log_write, "=" * 70)
            self.after(0, self._eval_log_write, f"Circuito: {cfg['circuit_name']}  |  Modo: {cfg['mode']}  |  Frontier: {cfg['frontier_mode']}")
            if cfg.get("routing_mask_config") is not None:
                mask_cfg = cfg["routing_mask_config"]
                self.after(
                    0,
                    self._eval_log_write,
                    "Mascara v3: "
                    f"cycle_window={mask_cfg.cycle_window}  |  "
                    f"stagnation_patience={mask_cfg.stagnation_patience}  |  "
                    f"sabre_top_k={mask_cfg.sabre_top_k}",
                )
            self.after(0, self._eval_log_write, f"Layout inicial: {eval_env.current_layout.tolist()}")
            self.after(0, self._eval_log_write, f"Puertas totales: {info['total_gates']}")
            if cfg["mode"] == "synthesis":
                self.after(0, self._eval_log_write, f"Residual symplectic: {obs['residual_symplectic'].tolist()}")
                self.after(0, self._eval_log_write, f"Residual phase: {obs['residual_phase'].tolist()}")
                self.after(0, self._eval_log_write, f"Physical->logical: {obs['physical_to_logical'].tolist()}")
            else:
                self.after(0, self._eval_log_write, f"Lookahead lógico: {obs['lookahead'].tolist()}")
                self.after(0, self._eval_log_write, f"Lookahead físico: {obs['lookahead_physical'].tolist()}")
                self.after(0, self._eval_log_write, f"Ejecutable: {obs['lookahead_executable'].tolist()}")
                self.after(0, self._eval_log_write, f"Distancia routing: {obs['lookahead_routing_distance'].tolist()}")
                self.after(0, self._eval_log_write, f"Máscara válida: {obs['lookahead_valid_mask'].tolist()}")
            self.after(0, self._eval_log_write, "-" * 70)

            total_reward = 0.0
            done = bool(info.get("already_completed_at_reset", False))
            step = 0
            cycle_detected = False
            should_detect_layout_cycles = cfg["mode"] == "routing"
            is_masked_routing = bool(cfg.get("masked"))

            # Detección de bucles (solo visual, no afecta al entorno de entrenamiento)
            from collections import Counter
            layout_visit_counts = Counter()
            if should_detect_layout_cycles:
                layout_visit_counts[tuple(eval_env.current_layout.tolist())] = 1
            CYCLE_THRESHOLD = 3  # Un layout visitado 3 veces = bucle claro

            while not done:
                layout_before = eval_env.current_layout.tolist()
                visible_frontier_before = []
                candidate_edges = []
                action_mask = []
                valid_action_indices = []
                if cfg["mode"] == "routing" and hasattr(eval_env, "get_visible_frontier_entries"):
                    visible_frontier_before = [
                        frontier_entry_to_dict(entry)
                        for entry in eval_env.get_visible_frontier_entries()
                    ]

                predict_kwargs = {}
                if is_masked_routing and cfg["mode"] == "routing":
                    raw_action_mask = np.asarray(eval_env.action_masks(), dtype=bool)
                    action_mask = raw_action_mask.tolist()
                    valid_action_indices = [
                        index for index, is_valid in enumerate(action_mask) if is_valid
                    ]
                    strategy = getattr(eval_env, "strategy", None)
                    candidate_edges = [tuple(edge) for edge in getattr(strategy, "edges", ())]
                    predict_kwargs["action_masks"] = raw_action_mask
                    self.after(0, self._eval_log_write, f"Candidate edges: {candidate_edges}")
                    self.after(0, self._eval_log_write, f"Action mask: {action_mask}")
                    self.after(0, self._eval_log_write, f"Valid action indices: {valid_action_indices}")

                action, _ = eval_agent.predict(obs, deterministic=True, **predict_kwargs)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                layout_after = eval_env.current_layout.tolist()

                total_reward += reward
                step += 1
                done = terminated or truncated

                # Detectar oscilación (solo afecta al display, no al entorno)
                if should_detect_layout_cycles:
                    current_layout_tuple = tuple(eval_env.current_layout.tolist())
                    layout_visit_counts[current_layout_tuple] += 1
                    if layout_visit_counts[current_layout_tuple] >= CYCLE_THRESHOLD:
                        cycle_detected = True
                        done = True

                record = EvaluationStepRecord(
                    step=step,
                    reward=float(reward),
                    action_type=info.get("action_type"),
                    is_valid_action=bool(info.get("is_valid_action", True)),
                    layout_before=list(layout_before),
                    layout_after=list(layout_after),
                    visible_frontier_before=visible_frontier_before,
                    executed_gates=list(info.get("executed_gates", [])),
                    swap_edge=info.get("swap_edge"),
                    routing_progress_delta=float(info.get("routing_progress_delta", 0.0)),
                    repeated_layout=bool(info.get("repeated_layout", False)),
                    undo_swap=bool(info.get("undo_swap", False)),
                    steps_without_progress=int(info.get("steps_without_progress", 0)),
                    stagnation_patience=info.get("stagnation_patience"),
                    truncation_reason=info.get("truncation_reason"),
                    primitive_name=info.get("primitive_name"),
                    primitive_physical_qargs=tuple(info.get("primitive_physical_qargs", ())),
                    primitive_cost=float(info.get("primitive_cost", 0.0)),
                    residual_distance_before=float(info.get("residual_distance_before", 0.0)),
                    residual_distance_after=float(info.get("residual_distance_after", 0.0)),
                    residual_distance_delta=float(info.get("residual_distance_delta", 0.0)),
                    candidate_edges=candidate_edges,
                    action_mask=action_mask,
                    valid_action_indices=valid_action_indices,
                )
                append_eval_record = getattr(self, "_append_eval_record", None)
                if not callable(append_eval_record):
                    append_eval_record = lambda eval_record: RLBenchmarkGUI._append_eval_record(self, eval_record)
                self.after(0, append_eval_record, record)

            self.after(0, self._eval_log_write, "-" * 70)

            if cycle_detected:
                status = "CICLO DETECTADO ⚠ (agente oscila sin avanzar)"
            elif info.get("is_completed") or info.get("already_completed_at_reset"):
                status = "COMPLETADO ✓"
            else:
                status = f"TRUNCADO ({info.get('truncation_reason') or 'max_steps'})"
            self.after(0, self._eval_log_write, f"\nResultado: {status}")
            self.after(0, self._eval_log_write, f"Steps totales: {step}")
            self.after(0, self._eval_log_write, f"SWAPs insertados: {eval_env.total_swaps}")
            self.after(0, self._eval_log_write, f"Reward acumulada: {total_reward:.1f}")
            self.after(0, self._eval_log_write, f"Layout final: {eval_env.current_layout.tolist()}")

        except Exception as e:
            self.after(0, self._eval_log_write, f"\n⚠ Error durante evaluación: {e}")
            import traceback
            self.after(0, self._eval_log_write, traceback.format_exc())

        finally:
            self.after(0, lambda: self._eval_button.configure(state="normal"))


# ===========================================================================
#  Punto de entrada
# ===========================================================================

if __name__ == "__main__":
    app = RLBenchmarkGUI()
    app.mainloop()
