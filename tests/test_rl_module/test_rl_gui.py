"""Focused regression tests for the RL GUI mode views."""

import importlib.util
import pathlib
import sys
import types

import numpy as np


def _install_stub_modules():
    repo_src_dir = pathlib.Path(__file__).resolve().parents[2] / "src"
    rl_module_dir = repo_src_dir / "rl_module"
    gui_dir = rl_module_dir / "gui"

    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *_args, **_kwargs: None
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.subplots = lambda *_args, **_kwargs: (None, [None, None])
    pyplot_stub.tight_layout = lambda *_args, **_kwargs: None
    pyplot_stub.close = lambda *_args, **_kwargs: None
    pyplot_stub.switch_backend = lambda *_args, **_kwargs: None
    backend_stub = types.ModuleType("matplotlib.backends.backend_tkagg")
    backend_stub.FigureCanvasTkAgg = object

    class _BaseWidget:
        def __init__(self, parent=None, **kwargs):
            self.parent = parent
            self.kwargs = dict(kwargs)
            self.text = kwargs.get("text")
            self.values = list(kwargs.get("values", [])) if "values" in kwargs else None
            self.command = kwargs.get("command")
            self.state = kwargs.get("state", "normal")
            self._value = None
            self.visible = False
            self.children = []
            if parent is not None and hasattr(parent, "children"):
                parent.children.append(self)

        def grid(self, **_kwargs):
            self.visible = True

        def grid_remove(self):
            self.visible = False

        def grid_columnconfigure(self, *_args, **_kwargs):
            return None

        def grid_rowconfigure(self, *_args, **_kwargs):
            return None

        def configure(self, **kwargs):
            self.kwargs.update(kwargs)
            if "text" in kwargs:
                self.text = kwargs["text"]
            if "state" in kwargs:
                self.state = kwargs["state"]
            if "values" in kwargs:
                self.values = list(kwargs["values"])

        def winfo_children(self):
            return list(self.children)

        def set(self, value):
            self._value = value
            if self.command is not None:
                self.command(value)

        def get(self):
            return self._value

    class _CTk(_BaseWidget):
        def title(self, _value):
            return None

        def geometry(self, _value):
            return None

        def grid_rowconfigure(self, *_args, **_kwargs):
            return None

    class _CTkFrame(_BaseWidget):
        pass

    class _CTkLabel(_BaseWidget):
        pass

    class _CTkOptionMenu(_BaseWidget):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent, **kwargs)
            if self.values:
                self._value = self.values[0]

    class _CTkSlider(_BaseWidget):
        pass

    class _CTkButton(_BaseWidget):
        pass

    class _CTkTabview(_BaseWidget):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent, **kwargs)
            self.tabs = {}

        def add(self, name):
            tab = _CTkFrame(self)
            self.tabs[name] = tab
            return tab

        def set(self, _value):
            return None

    class _CTkTextbox(_BaseWidget):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent, **kwargs)
            self.buffer = ""

        def insert(self, *_args, **_kwargs):
            if len(_args) >= 2:
                self.buffer += _args[1]
            return None

        def see(self, *_args, **_kwargs):
            return None

        def delete(self, *_args, **_kwargs):
            self.buffer = ""
            return None

    class _CTkProgressBar(_BaseWidget):
        pass

    ctk_stub = types.ModuleType("customtkinter")
    ctk_stub.CTk = _CTk
    ctk_stub.CTkFrame = _CTkFrame
    ctk_stub.CTkLabel = _CTkLabel
    ctk_stub.CTkOptionMenu = _CTkOptionMenu
    ctk_stub.CTkSlider = _CTkSlider
    ctk_stub.CTkButton = _CTkButton
    ctk_stub.CTkTabview = _CTkTabview
    ctk_stub.CTkTextbox = _CTkTextbox
    ctk_stub.CTkProgressBar = _CTkProgressBar
    ctk_stub.CTkFont = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    ctk_stub.set_appearance_mode = lambda *_args, **_kwargs: None
    ctk_stub.set_default_color_theme = lambda *_args, **_kwargs: None

    qiskit_stub = types.ModuleType("qiskit")

    class _QuantumCircuit:
        def __init__(self, qubits, name=None):
            self.num_qubits = qubits
            self.name = name

        def h(self, *_args, **_kwargs):
            return None

        def cx(self, *_args, **_kwargs):
            return None

    qiskit_stub.QuantumCircuit = _QuantumCircuit

    monitor_stub = types.ModuleType("stable_baselines3.common.monitor")
    monitor_stub.Monitor = object

    callbacks_stub = types.ModuleType("stable_baselines3.common.callbacks")

    class _BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose

    callbacks_stub.BaseCallback = _BaseCallback
    callbacks_stub.EvalCallback = object

    environment_stub = types.ModuleType("src.rl_module.environment")
    environment_stub.QuantumTranspilationEnv = object

    agent_stub = types.ModuleType("src.rl_module.agent")
    agent_stub.QuantumRLAgent = object

    training_stub = types.ModuleType("src.rl_module.training")
    training_stub.set_global_seeds = lambda *_args, **_kwargs: None

    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(repo_src_dir)]
    rl_module_pkg = types.ModuleType("src.rl_module")
    rl_module_pkg.__path__ = [str(rl_module_dir)]
    gui_pkg = types.ModuleType("src.rl_module.gui")
    gui_pkg.__path__ = [str(gui_dir)]

    sys.modules.setdefault("src", src_pkg)
    sys.modules.setdefault("src.rl_module", rl_module_pkg)
    sys.modules.setdefault("src.rl_module.gui", gui_pkg)
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)
    sys.modules.setdefault("matplotlib.backends.backend_tkagg", backend_stub)
    sys.modules.setdefault("customtkinter", ctk_stub)
    sys.modules.setdefault("qiskit", qiskit_stub)
    sys.modules.setdefault("stable_baselines3.common.monitor", monitor_stub)
    sys.modules.setdefault("stable_baselines3.common.callbacks", callbacks_stub)
    sys.modules.setdefault("src.rl_module.environment", environment_stub)
    sys.modules.setdefault("src.rl_module.agent", agent_stub)
    sys.modules.setdefault("src.rl_module.training", training_stub)

    return ctk_stub


def _load_gui_modules():
    _install_stub_modules()

    gui_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "rl_module" / "gui"
    loaded = {}
    for qualified_name, filename in [
        ("src.rl_module.gui.routing_view", "routing_view.py"),
        ("src.rl_module.gui.synthesis_view", "synthesis_view.py"),
        ("src.rl_module.gui.evaluation_panel", "evaluation_panel.py"),
        ("test_rl_gui_module", "rl_gui.py"),
    ]:
        sys.modules.pop(qualified_name, None)
        module_path = gui_dir / filename
        spec = importlib.util.spec_from_file_location(qualified_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        loaded[qualified_name] = module
    return loaded


def _load_gui_module(qualified_name, filename):
    _install_stub_modules()

    gui_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "rl_module" / "gui"
    sys.modules.pop(qualified_name, None)
    module_path = gui_dir / filename
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_MODULES = _load_gui_modules()
RL_GUI_MODULE = _MODULES["test_rl_gui_module"]
ROUTING_VIEW_MODULE = _MODULES["src.rl_module.gui.routing_view"]
SYNTHESIS_VIEW_MODULE = _MODULES["src.rl_module.gui.synthesis_view"]
EVALUATION_PANEL_MODULE = _MODULES["src.rl_module.gui.evaluation_panel"]
RLBenchmarkGUI = RL_GUI_MODULE.RLBenchmarkGUI
RoutingView = ROUTING_VIEW_MODULE.RoutingView
SynthesisView = SYNTHESIS_VIEW_MODULE.SynthesisView
EvaluationStepRecord = EVALUATION_PANEL_MODULE.EvaluationStepRecord
SYNTHESIS_BASIS_PROFILES = RL_GUI_MODULE.SYNTHESIS_BASIS_PROFILES


class TestRLGuiModeViews:
    def test_create_sidebar_builds_real_mode_views_and_routing_mode_shows_routing_controls(self):
        app = RLBenchmarkGUI()

        assert isinstance(app._routing_view, RoutingView)
        assert isinstance(app._synthesis_view, SynthesisView)
        assert app._routing_view.visible is True
        assert app._synthesis_view.visible is False
        assert app._frontier_option.parent is app._routing_view
        assert app._lookahead_slider.parent is app._routing_view
        assert app._basis_profile_option.parent is app._synthesis_view
        assert app._routing_view.frontier_option.visible is True
        assert app._routing_view.lookahead_slider.visible is True

    def test_switching_to_synthesis_shows_basis_controls_and_not_routing_primary_controls(self):
        app = RLBenchmarkGUI()

        app._on_mode_changed("synthesis")

        assert app._routing_view.visible is False
        assert app._synthesis_view.visible is True
        assert app._basis_profile_option.parent is app._synthesis_view
        assert app._basis_profile_option.visible is True
        assert app._routing_view.frontier_option.visible is False
        assert app._routing_view.lookahead_slider.visible is False
        visible_primary_texts = [
            child.text
            for child in app._synthesis_view.children
            if getattr(child, "visible", False) and getattr(child, "text", None)
        ]
        assert "Synthesis Basis:" in visible_primary_texts
        assert "Frontier:" not in visible_primary_texts
        assert not any(text.startswith("Lookahead:") for text in visible_primary_texts)

    def test_get_config_preserves_shared_training_fields_and_uses_basis_gates_for_synthesis(self):
        app = RLBenchmarkGUI()

        app._mode_option.set("synthesis")
        app._basis_profile_option.set("Clifford-CZ")
        app._frontier_option.set("dag")
        app._algo_option.set("PPO")
        app._timesteps_slider.set(1234)
        app._maxsteps_slider.set(321)
        app._lookahead_slider.set(7)
        app._seed_slider.set(9)

        cfg = app._get_config()

        assert cfg["mode"] == "synthesis"
        assert cfg["basis_gates"] == SYNTHESIS_BASIS_PROFILES["Clifford-CZ"]
        assert cfg["frontier_mode"] == "dag"
        assert cfg["algorithm"] == "PPO"
        assert cfg["timesteps"] == 1234
        assert cfg["max_steps"] == 321
        assert cfg["lookahead"] == 7
        assert cfg["seed"] == 9


class TestRLEvaluationInspector:
    def test_episode_inspector_panel_renders_routing_summary_and_details(self):
        panel = EVALUATION_PANEL_MODULE.EpisodeInspectorPanel(parent=None)
        record = EvaluationStepRecord(
            step=3,
            reward=1.25,
            action_type="swap",
            is_valid_action=True,
            layout_before=[0, 1, 2],
            layout_after=[1, 0, 2],
            visible_frontier_before=[
                {
                    "gate_name": "cx",
                    "logical_q1": 0,
                    "logical_q2": 2,
                    "physical_q1": 1,
                    "physical_q2": 2,
                    "executable": False,
                }
            ],
            executed_gates=[("cx", 0, 2)],
            swap_edge=(0, 1),
            routing_progress_delta=1.0,
            repeated_layout=True,
            undo_swap=True,
        )

        panel.set_records([record])

        assert "Resumen routing:" in panel._details.buffer
        assert "swap_edge: (0, 1)" in panel._details.buffer
        assert "executed_gates: [('cx', 0, 2)]" in panel._details.buffer
        assert "visible_frontier_before:" in panel._details.buffer
        assert "gate_name=cx" in panel._details.buffer
        assert "repeated_layout: True" in panel._details.buffer
        assert "undo_swap: True" in panel._details.buffer

    def test_episode_inspector_panel_renders_synthesis_summary_and_details(self):
        panel = EVALUATION_PANEL_MODULE.EpisodeInspectorPanel(parent=None)
        record = EvaluationStepRecord(
            step=4,
            reward=0.75,
            action_type="gate",
            is_valid_action=True,
            layout_before=[1, 0],
            layout_after=[1, 0],
            primitive_name="cz",
            primitive_physical_qargs=(0, 1),
            primitive_cost=2.0,
            residual_distance_before=5.0,
            residual_distance_after=2.0,
            residual_distance_delta=3.0,
        )

        panel.set_records([record])

        assert "Resumen synthesis:" in panel._details.buffer
        assert "primitive_name: cz" in panel._details.buffer
        assert "primitive_physical_qargs: (0, 1)" in panel._details.buffer
        assert "primitive_cost: 2.0" in panel._details.buffer
        assert "residual progression: 5.0 -> 2.0 (delta +3.000)" in panel._details.buffer

    def test_episode_inspector_panel_synthesis_summary_avoids_routing_only_labels(self):
        panel = EVALUATION_PANEL_MODULE.EpisodeInspectorPanel(parent=None)
        record = EvaluationStepRecord(
            step=5,
            reward=0.5,
            action_type="gate",
            is_valid_action=True,
            layout_before=[0, 1],
            layout_after=[0, 1],
            primitive_name="iswap",
            primitive_physical_qargs=(0, 1),
            primitive_cost=3.0,
            residual_distance_before=6.0,
            residual_distance_after=1.0,
            residual_distance_delta=5.0,
        )

        panel.set_records([record])

        assert "Resumen synthesis:" in panel._details.buffer
        assert "SWAPs insertados" not in panel._details.buffer

    def test_evaluation_tab_uses_shared_inspector_panel_and_updates_when_records_arrive(self):
        gui_module = _load_gui_module("test_rl_gui_eval_module", "rl_gui.py")
        app = gui_module.RLBenchmarkGUI()

        record = EvaluationStepRecord(
            step=1,
            reward=1.5,
            action_type="swap",
            is_valid_action=True,
            layout_before=[0, 1, 2],
            layout_after=[1, 0, 2],
            visible_frontier_before=[
                {
                    "gate_name": "cx",
                    "logical_q1": 0,
                    "logical_q2": 2,
                    "physical_q1": 0,
                    "physical_q2": 2,
                    "executable": False,
                }
            ],
            executed_gates=[("cx", 0, 1)],
            swap_edge=(0, 1),
            routing_progress_delta=1.0,
            repeated_layout=False,
            undo_swap=False,
        )

        assert isinstance(app._eval_inspector, EVALUATION_PANEL_MODULE.EpisodeInspectorPanel)
        assert app._eval_inspector.selected_record is None

        app._append_eval_record(record)

        assert app._eval_log == [record]
        assert app._eval_inspector.records == [record]
        assert app._eval_inspector.selected_record == record

    def test_episode_inspector_panel_renders_selected_record_details(self):
        panel = EVALUATION_PANEL_MODULE.EpisodeInspectorPanel(parent=None)
        record = EvaluationStepRecord(
            step=2,
            reward=-0.5,
            action_type="gate",
            is_valid_action=False,
            layout_before=[1],
            layout_after=[1],
            visible_frontier_before=[{"gate_name": "cz", "logical_q1": 0}],
            executed_gates=[("cz", 0, 0)],
            primitive_name="cz",
            primitive_physical_qargs=(0, 1),
            primitive_cost=2.0,
            residual_distance_before=4.0,
            residual_distance_after=1.0,
            residual_distance_delta=3.0,
        )

        panel.set_records([record])

        assert panel.selected_record == record
        assert "Paso: 2" in panel._details.buffer
        assert "Accion: gate" in panel._details.buffer
        assert "Resumen synthesis:" in panel._details.buffer
        assert "primitive_name: cz" in panel._details.buffer
        assert "residual progression: 4.0 -> 1.0 (delta +3.000)" in panel._details.buffer

    def test_episode_inspector_panel_can_browse_multiple_records(self):
        panel = EVALUATION_PANEL_MODULE.EpisodeInspectorPanel(parent=None)
        first_record = EvaluationStepRecord(
            step=1,
            reward=1.0,
            action_type="swap",
            is_valid_action=True,
            layout_before=[0, 1],
            layout_after=[1, 0],
            swap_edge=(0, 1),
        )
        second_record = EvaluationStepRecord(
            step=2,
            reward=-0.5,
            action_type="gate",
            is_valid_action=False,
            layout_before=[1, 0],
            layout_after=[1, 0],
            primitive_name="cz",
            primitive_physical_qargs=(0, 1),
            primitive_cost=2.0,
            residual_distance_before=4.0,
            residual_distance_after=1.0,
            residual_distance_delta=3.0,
        )

        panel.set_records([first_record, second_record])

        assert panel.selected_record == second_record
        assert panel._step_selector.values == ["Paso 1", "Paso 2"]
        panel._step_selector.set("Paso 1")

        assert panel.selected_record == first_record
        assert "Paso: 1" in panel._details.buffer
        assert "Resumen routing:" in panel._details.buffer
        assert "swap_edge: (0, 1)" in panel._details.buffer
        assert "primitive_name: cz" not in panel._details.buffer

    def test_episode_inspector_panel_renders_both_sections_when_record_has_mixed_metadata(self):
        panel = EVALUATION_PANEL_MODULE.EpisodeInspectorPanel(parent=None)
        record = EvaluationStepRecord(
            step=6,
            reward=1.0,
            action_type="gate",
            is_valid_action=True,
            layout_before=[0, 1],
            layout_after=[1, 0],
            visible_frontier_before=[
                {
                    "gate_name": "cx",
                    "logical_q1": 0,
                    "logical_q2": 1,
                    "physical_q1": 0,
                    "physical_q2": 1,
                    "executable": True,
                }
            ],
            executed_gates=[("cx", 0, 1)],
            swap_edge=(0, 1),
            routing_progress_delta=0.5,
            repeated_layout=False,
            undo_swap=False,
            primitive_name="cz",
            primitive_physical_qargs=(0, 1),
            primitive_cost=2.0,
            residual_distance_before=3.0,
            residual_distance_after=1.0,
            residual_distance_delta=2.0,
        )

        panel.set_records([record])

        assert "Resumen routing:" in panel._details.buffer
        assert "Detalles routing:" in panel._details.buffer
        assert "Resumen synthesis:" in panel._details.buffer
        assert "Detalles synthesis:" in panel._details.buffer
        assert "swap_edge: (0, 1)" in panel._details.buffer
        assert "primitive_name: cz" in panel._details.buffer

    def test_episode_inspector_panel_preserves_selection_when_new_records_arrive(self):
        panel = EVALUATION_PANEL_MODULE.EpisodeInspectorPanel(parent=None)
        first_record = EvaluationStepRecord(
            step=1,
            reward=1.0,
            action_type="swap",
            is_valid_action=True,
            layout_before=[0],
            layout_after=[0],
        )
        second_record = EvaluationStepRecord(
            step=2,
            reward=2.0,
            action_type="swap",
            is_valid_action=True,
            layout_before=[0],
            layout_after=[0],
        )
        third_record = EvaluationStepRecord(
            step=3,
            reward=3.0,
            action_type="gate",
            is_valid_action=True,
            layout_before=[0],
            layout_after=[0],
            primitive_name="x",
        )

        panel.set_records([first_record, second_record])
        panel._step_selector.set("Paso 1")
        panel.set_records([first_record, second_record, third_record])

        assert panel.selected_record == first_record
        assert panel._step_selector.get() == "Paso 1"
        assert "Paso: 1" in panel._details.buffer

    def test_start_evaluation_clears_inspector_state_before_launching_thread(self):
        gui_module = _load_gui_module("test_rl_gui_start_eval_module", "rl_gui.py")
        app = gui_module.RLBenchmarkGUI()
        app._agent = object()
        app._is_training = False
        app._eval_log = [
            EvaluationStepRecord(
                step=1,
                reward=1.0,
                action_type="swap",
                is_valid_action=True,
                layout_before=[0],
                layout_after=[0],
            )
        ]
        app._eval_inspector.set_records(app._eval_log)

        original_thread = gui_module.threading.Thread

        class DummyThread:
            def __init__(self, target=None, daemon=None):
                self.target = target
                self.daemon = daemon

            def start(self):
                return None

        gui_module.threading.Thread = DummyThread
        try:
            app._start_evaluation()
        finally:
            gui_module.threading.Thread = original_thread

        assert app._eval_log == []
        assert app._eval_inspector.records == []
        assert app._eval_inspector.selected_record is None
        assert app._eval_terminal.buffer == ""

    def test_evaluation_thread_builds_structured_routing_step_records(self):
        rl_gui = _load_gui_module("test_rl_gui_eval_routing_module", "rl_gui.py")
        eval_log_lines = []

        class DummyEvalEnv:
            def __init__(self, *args, **kwargs):
                self.current_layout = np.array([0, 1, 2], dtype=np.int32)
                self.remaining_gates = [("cx", 0, 2)]
                self.total_swaps = 0

            def get_visible_frontier_entries(self):
                return [
                    types.SimpleNamespace(
                        gate_name="cx",
                        logical_q1=0,
                        logical_q2=2,
                        physical_q1=0,
                        physical_q2=2,
                        executable=False,
                    )
                ]

            def reset(self, seed=None):
                obs = {
                    "lookahead": np.array([0, 2], dtype=np.int32),
                    "lookahead_physical": np.array([0, 2], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([1.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([1.0], dtype=np.float32),
                }
                return obs, {"total_gates": 1}

            def step(self, action):
                self.current_layout = np.array([1, 0, 2], dtype=np.int32)
                self.remaining_gates = []
                self.total_swaps = 1
                obs = {
                    "lookahead": np.array([-1, -1], dtype=np.int32),
                    "lookahead_physical": np.array([-1, -1], dtype=np.int32),
                    "lookahead_executable": np.array([0.0], dtype=np.float32),
                    "lookahead_routing_distance": np.array([0.0], dtype=np.float32),
                    "lookahead_valid_mask": np.array([0.0], dtype=np.float32),
                }
                info = {
                    "action_type": "swap",
                    "is_valid_action": True,
                    "gates_executed": 1,
                    "executed_gates": [("cx", 0, 2)],
                    "swap_edge": (0, 1),
                    "routing_progress_delta": 1.0,
                    "repeated_layout": False,
                    "undo_swap": False,
                    "is_completed": True,
                }
                return obs, 2.5, True, False, info

        class DummyAgent:
            def predict(self, observation, deterministic=True):
                return 0, None

        class DummyButton:
            def configure(self, **kwargs):
                return None

        class DummyTabView:
            def set(self, tab_name):
                return None

        class DummyGUI:
            def __init__(self):
                self._is_training = False
                self._agent = DummyAgent()
                self._eval_log = []
                self._eval_inspector = types.SimpleNamespace(records=[])
                self._training_cfg = {
                    "seed": 42,
                    "circuit": object(),
                    "circuit_name": "fixture-routing-structured",
                    "coupling_map": [(0, 1), (1, 2)],
                    "mode": "routing",
                    "frontier_mode": "sequential",
                    "lookahead": 1,
                    "max_steps": 5,
                    "algorithm": "PPO",
                    "best_model_path": None,
                    "last_model_path": None,
                    "run_model_dir": "models",
                }
                self._eval_button = DummyButton()
                self._tabview = DummyTabView()

            def _append_eval_record(self, record):
                self._eval_log.append(record)
                self._eval_inspector.records = list(self._eval_log)

            def _get_config(self):
                return self._training_cfg

            def _clear_eval_terminal(self):
                return None

            def _eval_log_write(self, text):
                eval_log_lines.append(text)

            def after(self, _delay, callback, *args):
                callback(*args)

        rl_gui.set_global_seeds = lambda seed: None
        rl_gui.QuantumTranspilationEnv = DummyEvalEnv

        gui = DummyGUI()
        rl_gui.RLBenchmarkGUI._evaluation_thread(gui)

        assert len(gui._eval_log) == 1
        record = gui._eval_log[0]
        assert record.step == 1
        assert record.reward == 2.5
        assert record.action_type == "swap"
        assert record.is_valid_action is True
        assert record.layout_before == [0, 1, 2]
        assert record.layout_after == [1, 0, 2]
        assert record.visible_frontier_before == [
            {
                "gate_name": "cx",
                "logical_q1": 0,
                "logical_q2": 2,
                "physical_q1": 0,
                "physical_q2": 2,
                "executable": False,
            }
        ]
        assert record.executed_gates == [("cx", 0, 2)]
        assert record.swap_edge == (0, 1)
        assert record.routing_progress_delta == 1.0
        assert record.repeated_layout is False
        assert record.undo_swap is False
        assert gui._eval_inspector.records == [record]
        assert not any("│" in line for line in eval_log_lines)

    def test_evaluation_thread_builds_structured_synthesis_step_records(self):
        rl_gui = _load_gui_module("test_rl_gui_eval_synthesis_module", "rl_gui.py")

        class DummyEvalEnv:
            def __init__(self, *args, **kwargs):
                self.current_layout = np.array([1], dtype=np.int32)
                self.remaining_gates = ["pending"]
                self.total_swaps = 0

            def reset(self, seed=None):
                obs = {
                    "layout": np.array([1], dtype=np.int32),
                    "physical_to_logical": np.array([-1, 0], dtype=np.int32),
                    "residual_symplectic": np.zeros(16, dtype=np.int32),
                    "residual_phase": np.zeros(4, dtype=np.int32),
                    "step_progress": np.array([0.0], dtype=np.float32),
                }
                return obs, {"total_gates": 1}

            def step(self, action):
                self.remaining_gates = []
                obs = {
                    "layout": np.array([1], dtype=np.int32),
                    "physical_to_logical": np.array([-1, 0], dtype=np.int32),
                    "residual_symplectic": np.zeros(16, dtype=np.int32),
                    "residual_phase": np.zeros(4, dtype=np.int32),
                    "step_progress": np.array([1.0], dtype=np.float32),
                }
                info = {
                    "action_type": "gate",
                    "is_valid_action": True,
                    "gates_executed": 0,
                    "primitive_name": "cz",
                    "primitive_physical_qargs": (0, 1),
                    "primitive_cost": 2.0,
                    "residual_distance_before": 5,
                    "residual_distance_after": 2,
                    "residual_distance_delta": 3.0,
                    "is_completed": True,
                }
                return obs, 1.25, True, False, info

        class DummyAgent:
            def predict(self, observation, deterministic=True):
                return 0, None

        class DummyButton:
            def configure(self, **kwargs):
                return None

        class DummyTabView:
            def set(self, tab_name):
                return None

        class DummyGUI:
            def __init__(self):
                self._is_training = False
                self._agent = DummyAgent()
                self._eval_log = []
                self._eval_inspector = types.SimpleNamespace(records=[])
                self._training_cfg = {
                    "seed": 42,
                    "circuit": object(),
                    "circuit_name": "fixture-synthesis-structured",
                    "coupling_map": [(0, 1)],
                    "mode": "synthesis",
                    "basis_gates": ["cz", "rz", "sx", "x"],
                    "frontier_mode": "sequential",
                    "lookahead": 1,
                    "max_steps": 5,
                    "algorithm": "PPO",
                    "best_model_path": None,
                    "last_model_path": None,
                    "run_model_dir": "models",
                }
                self._eval_button = DummyButton()
                self._tabview = DummyTabView()

            def _append_eval_record(self, record):
                self._eval_log.append(record)
                self._eval_inspector.records = list(self._eval_log)

            def _get_config(self):
                return self._training_cfg

            def _clear_eval_terminal(self):
                return None

            def _eval_log_write(self, text):
                return None

            def after(self, _delay, callback, *args):
                callback(*args)

        rl_gui.set_global_seeds = lambda seed: None
        rl_gui.QuantumTranspilationEnv = DummyEvalEnv

        gui = DummyGUI()
        rl_gui.RLBenchmarkGUI._evaluation_thread(gui)

        assert len(gui._eval_log) == 1
        record = gui._eval_log[0]
        assert record.step == 1
        assert record.reward == 1.25
        assert record.action_type == "gate"
        assert record.is_valid_action is True
        assert record.layout_before == [1]
        assert record.layout_after == [1]
        assert record.primitive_name == "cz"
        assert record.primitive_physical_qargs == (0, 1)
        assert record.primitive_cost == 2.0
        assert record.residual_distance_before == 5
        assert record.residual_distance_after == 2
        assert record.residual_distance_delta == 3.0
        assert gui._eval_inspector.records == [record]
