"""Focused regression tests for the RL GUI mode views."""

import importlib.util
import pathlib
import sys
import types


def _install_stub_modules():
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda *_args, **_kwargs: None
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    pyplot_stub.subplots = lambda *_args, **_kwargs: (None, [None, None])
    pyplot_stub.tight_layout = lambda *_args, **_kwargs: None
    pyplot_stub.close = lambda *_args, **_kwargs: None
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
        def insert(self, *_args, **_kwargs):
            return None

        def see(self, *_args, **_kwargs):
            return None

        def delete(self, *_args, **_kwargs):
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
    src_pkg.__path__ = []
    rl_module_pkg = types.ModuleType("src.rl_module")
    rl_module_pkg.__path__ = []
    gui_pkg = types.ModuleType("src.rl_module.gui")
    gui_pkg.__path__ = []

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


_MODULES = _load_gui_modules()
RL_GUI_MODULE = _MODULES["test_rl_gui_module"]
ROUTING_VIEW_MODULE = _MODULES["src.rl_module.gui.routing_view"]
SYNTHESIS_VIEW_MODULE = _MODULES["src.rl_module.gui.synthesis_view"]
RLBenchmarkGUI = RL_GUI_MODULE.RLBenchmarkGUI
RoutingView = ROUTING_VIEW_MODULE.RoutingView
SynthesisView = SYNTHESIS_VIEW_MODULE.SynthesisView
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
