"""Tests de regresion de la GUI de benchmark/tuning."""

import tkinter as tk
import types
from queue import Queue

import pytest

from src.mo_module.benchmark.benchmark_gui import BenchmarkGUI


def _require_usable_tk():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk runtime is not available in this Python environment: {exc}")
    else:
        root.withdraw()
        root.destroy()


class _DummyWidget:
    def __init__(self, **values):
        self.values = dict(values)
        self.calls = []

    def configure(self, **kwargs):
        self.values.update(kwargs)
        self.calls.append(("configure", kwargs))

    def set(self, value):
        self.values["value"] = value
        self.calls.append(("set", value))

    def get(self):
        return self.values.get("value")

    def delete(self, *_args):
        self.calls.append(("delete", _args))

    def destroy(self):
        self.calls.append(("destroy", None))


class _DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _DummyContainer:
    def winfo_children(self):
        return []


class _DummyTabView:
    def __init__(self):
        self.selected = None

    def set(self, value):
        self.selected = value


class _DummyGuiHarness:
    def __init__(self):
        self._tuning_event_queue = Queue()
        self._tuning_poll_after_id = None
        self._tuning_poll_shutdown = False
        self._tuning_run_succeeded = False
        self._window_exists = True
        self._tuner = object()
        self._active_manual_ref_point = None
        self.is_tuning = True
        self.t_ref_point_mode = _DummyVar("calibrated")
        self.t_progress_bar = _DummyWidget(value=0)
        self.t_progress_label = _DummyWidget(text="")
        self.t_phase_value = _DummyWidget(text="")
        self.t_trial_value = _DummyWidget(text="")
        self.t_best_value = _DummyWidget(text="")
        self.t_ref_value = _DummyWidget(text="")
        self.t_run_button = _DummyWidget(state="disabled")
        self.t_best_config_btn = _DummyWidget(state="disabled")
        self.logged = []
        self.after_calls = []
        self.after_cancel_calls = []
        self.state_changes = []
        self.plot_render_calls = []
        self.destroy_called = False

    def after(self, delay, callback):
        self.after_calls.append((delay, callback))
        self._tuning_poll_after_id = f"after-{len(self.after_calls)}"
        return self._tuning_poll_after_id

    def after_cancel(self, after_id):
        self.after_cancel_calls.append(after_id)
        if self._tuning_poll_after_id == after_id:
            self._tuning_poll_after_id = None

    def destroy(self):
        self._window_exists = False
        self.destroy_called = True
        BenchmarkGUI._stop_tuning_event_polling(self)

    def _poll_tuning_events(self):
        return BenchmarkGUI._poll_tuning_events(self)

    def winfo_exists(self):
        return self._window_exists

    def t_log(self, text):
        self.logged.append(text)

    def _set_tuning_input_state(self, enabled):
        self.state_changes.append(enabled)

    def _render_tuning_plots(self):
        self.plot_render_calls.append(
            (self.state_changes[:], self.t_run_button.values["state"], self.is_tuning)
        )


class TestBenchmarkGuiInitialization:
    """Regresiones de inicializacion y layout de la GUI."""

    def test_gui_initialization_sets_tuning_state_before_building_controls(self):
        """BenchmarkGUI arranca sin delegar en __getattr__ de Tk para is_tuning."""
        _require_usable_tk()
        app = BenchmarkGUI()
        try:
            assert app.is_tuning is False
            assert app._tuner is None
        finally:
            app.destroy()

    def test_tuning_start_button_is_visible_inside_sidebar(self):
        """El boton de iniciar tuning debe quedar dentro del alto visible del sidebar."""
        _require_usable_tk()
        app = BenchmarkGUI()
        try:
            app.master_tabview.set("Modo Tuning (Optuna)")
            app.update()
            app.update_idletasks()
            visible_bottom = app.t_sidebar.winfo_height()
            run_button_bottom = app.t_run_button.winfo_y() + app.t_run_button.winfo_height()

            assert run_button_bottom <= visible_bottom
        finally:
            app.destroy()


class TestBenchmarkGuiTuningQueue:
    """Regresiones del flujo de cola entre worker y main thread."""

    def test_worker_progress_callback_enqueues_events_without_touching_tk(self):
        """El worker solo publica eventos en cola y no usa after/Tk directamente."""
        harness = _DummyGuiHarness()
        event = {"event": "trial_completed", "completed_trials": 1, "total_trials": 3}

        BenchmarkGUI._handle_tuner_progress_event(harness, event)

        queued = harness._tuning_event_queue.get_nowait()
        assert queued == {"type": "progress", "payload": event}
        assert harness.after_calls == []

    def test_calibration_progress_uses_existing_step_fields_for_visible_warmup_state(self):
        """La GUI traduce current_step/total_steps a texto y barra visibles de warm-up."""
        # Arrange
        harness = _DummyGuiHarness()
        progress_event = {
            "event": "calibration_progress",
            "current_step": 2,
            "total_steps": 3,
            "ref_point_mode": "calibrated",
            "ref_point_candidate": [8.8, 9.9],
            "config": {
                "algorithm": "nsga2",
                "population_size": 30,
                "n_generations": 50,
            },
        }

        # Act
        BenchmarkGUI._apply_tuner_progress_event(harness, progress_event)

        # Assert
        assert harness.t_progress_label.values["text"] == "Warm-up 2/3 completado (calibrated)."
        assert harness.t_phase_value.values["text"] == "Fase: warm-up"
        assert harness.t_trial_value.values["text"] == "Warm-up: 2/3"
        assert harness.t_progress_bar.values["value"] == 0.1 * (2 / 3)
        assert harness.t_ref_value.values["text"] == "Ref. point: calibrated [8.800, 9.900]"
        assert any("[Warm-up 2/3]" in line for line in harness.logged)

    def test_polling_loop_processes_completed_event_and_restores_ui_before_plotting(self):
        """La cola usa completed y decide el render en el main thread tras restaurar la UI."""
        # Arrange
        harness = _DummyGuiHarness()
        calibration_progress_event = {
            "type": "progress",
            "payload": {
                "event": "calibration_progress",
                "current_step": 2,
                "total_steps": 3,
                "ref_point_mode": "calibrated",
                "ref_point_candidate": [8.8, 9.9],
                "config": {
                    "algorithm": "nsga2",
                    "population_size": 30,
                    "n_generations": 50,
                },
            },
        }
        completed_event = {
            "type": "completed",
            "elapsed_s": 12.3,
            "summary": "summary text",
            "best_config": "best config text",
            "best_score": 0.8,
            "total_trials": 5,
            "ref_point_mode": "calibrated",
            "ref_point": [8.8, 9.9],
            "can_copy_best_config": True,
        }

        harness._tuning_event_queue.put(calibration_progress_event)
        harness._tuning_event_queue.put(completed_event)

        # Act
        BenchmarkGUI._poll_tuning_events(harness)

        # Assert
        assert harness.t_phase_value.values["text"] == "Fase: completado"
        assert harness.t_progress_label.values["text"] == "Tuning completado (calibrated)."
        assert harness.t_progress_bar.values["value"] == 1
        assert harness.t_trial_value.values["text"] == "Trial: 5/5"
        assert harness.t_best_value.values["text"] == "Best HV: 0.800000"
        assert harness._tuning_run_succeeded is True
        assert harness.state_changes == [True]
        assert harness.t_run_button.values["state"] == "normal"
        assert harness.t_best_config_btn.values["state"] == "normal"
        assert harness.t_ref_value.values["text"] == "Ref. point: calibrated [8.800, 9.900]"
        assert harness.is_tuning is False
        assert harness.plot_render_calls == [([True], "normal", False)]

    def test_error_event_sets_authoritative_error_labels_and_clears_success_state(self):
        """El finalizador de error limpia el estado de exito y deja etiquetas terminales coherentes."""
        harness = _DummyGuiHarness()
        harness._tuning_run_succeeded = True
        harness.t_progress_bar.values["value"] = 1
        harness.t_progress_label.values["text"] = "Tuning completado (calibrated)."
        harness.t_phase_value.values["text"] = "Fase: completado"
        harness.t_trial_value.values["text"] = "Trial: 5/5"
        harness.t_best_value.values["text"] = "Best HV: 0.800000"
        harness.t_ref_value.values["text"] = "Ref. point: calibrated [8.800, 9.900]"

        BenchmarkGUI._finalize_tuning_error(
            harness,
            {
                "type": "error",
                "message": "boom",
                "ref_point_mode": "calibrated",
            },
        )

        assert harness._tuning_run_succeeded is False
        assert harness.t_progress_bar.values["value"] == 0
        assert harness.t_progress_label.values["text"] == "Tuning interrumpido por error."
        assert harness.t_phase_value.values["text"] == "Fase: error"
        assert harness.t_trial_value.values["text"] == "Trial: 0/0"
        assert harness.t_best_value.values["text"] == "Best HV: -"
        assert harness.t_ref_value.values["text"] == "Ref. point: calibrated pending (warm-up automatic)"
        assert harness.t_best_config_btn.values["state"] == "disabled"
        assert harness.plot_render_calls == []

    def test_polling_loop_reschedules_while_tuning_is_active(self):
        """Mientras siga activo, el poller vuelve a programarse desde el main thread."""
        harness = _DummyGuiHarness()

        BenchmarkGUI._poll_tuning_events(harness)

        assert harness.after_calls == [(50, harness._poll_tuning_events)]
        assert harness._tuning_poll_after_id == "after-1"

    def test_destroy_cancels_pending_tuning_poll_and_blocks_reschedule(self):
        """Cerrar la ventana cancela el after pendiente y marca shutdown del polling."""
        harness = _DummyGuiHarness()
        harness._tuning_poll_after_id = "after-7"

        harness.destroy()

        assert harness._tuning_poll_shutdown is True
        assert harness.after_cancel_calls == ["after-7"]
        assert harness._tuning_poll_after_id is None

    def test_polling_loop_does_not_reschedule_after_shutdown(self):
        """Si la GUI entra en shutdown, el polling no deja callbacks vivos."""
        harness = _DummyGuiHarness()
        harness._tuning_poll_shutdown = True

        BenchmarkGUI._poll_tuning_events(harness)

        assert harness.after_calls == []

    def test_polling_loop_does_not_touch_widgets_after_window_is_gone(self):
        """Si la ventana ya no existe, el poller sale sin procesar ni reprogramar."""
        harness = _DummyGuiHarness()
        harness._window_exists = False
        harness._tuning_event_queue.put(
            {
                "type": "progress",
                "payload": {
                    "event": "trial_completed",
                    "completed_trials": 1,
                    "total_trials": 3,
                    "score": 0.2,
                    "best_score": 0.2,
                    "params": {},
                },
            }
        )

        BenchmarkGUI._poll_tuning_events(harness)

        assert harness.t_progress_label.values["text"] == ""
        assert harness.after_calls == []

    def test_copy_best_to_benchmark_preserves_tuned_crossover_for_benchmark_run(
        self, monkeypatch
    ):
        """Copiar la mejor config debe propagar el crossover real al benchmark."""
        # Arrange
        import src.mo_module.benchmark.benchmark_gui as gui_module

        captured_thread = {}

        class _FakeThread:
            def __init__(self, *, target, args, daemon):
                captured_thread["target"] = target
                captured_thread["args"] = args
                captured_thread["daemon"] = daemon

            def start(self):
                captured_thread["started"] = True

        monkeypatch.setattr(gui_module.threading, "Thread", _FakeThread)

        best_config = types.SimpleNamespace(
            population_size=24,
            n_generations=12,
            prob_swap_mutation=0.5,
            prob_replace_mutation=0.9,
            algorithm="nsga2",
            crossover_operator="ox",
        )

        class _DummyTuner:
            def best_config(self):
                return best_config

        harness = types.SimpleNamespace(
            _tuning_run_succeeded=True,
            _tuner=_DummyTuner(),
            pop_slider=_DummyWidget(value=30),
            gens_slider=_DummyWidget(value=50),
            swap_mut_option=_DummyWidget(value="0.3"),
            replace_mut_option=_DummyWidget(value="0.7"),
            algo_option=_DummyWidget(value="moead"),
            master_tabview=_DummyTabView(),
            is_running=False,
            run_button=_DummyWidget(state="normal"),
            terminal_text=_DummyWidget(state="disabled"),
            summary_text=_DummyWidget(state="disabled"),
            plot_frame=_DummyContainer(),
            circuit_vars={gui_module.DEFAULT_BENCHMARK_CIRCUITS[0].name: _DummyVar(True)},
            seeds_slider=_DummyWidget(value=1),
            workers_slider=_DummyWidget(value=1),
            backend_option=_DummyWidget(value="fake_torino"),
            log=lambda _text: None,
            t_log=lambda _text: None,
            _update_pop_label=lambda _value: None,
            _update_gens_label=lambda _value: None,
            _update_swap_mut_label=lambda _value: None,
            _update_replace_mut_label=lambda _value: None,
            _run_benchmark_thread=lambda *args: None,
        )

        # Act
        BenchmarkGUI._copy_best_to_benchmark(harness)
        BenchmarkGUI.start_benchmark(harness)

        # Assert
        assert harness._benchmark_crossover_operator == "ox"
        assert captured_thread["target"] == harness._run_benchmark_thread
        assert captured_thread["args"][7] == "ox"
        assert captured_thread["started"] is True


class TestRenderTuningPlots:
    """Regresiones del método _render_tuning_plots."""

    def test_render_tuning_plots_completes_without_ax_argument_error(self, monkeypatch):
        """_render_tuning_plots no pasa 'ax=' a plot_optimization_history (bug regresión).

        Optuna 4.x no admite ax= en plot_optimization_history; la función crea su
        propio subplot y devuelve el Axes. El método debe llamarla sin ese argumento,
        obtener el Axes devuelto y extraer la figura con ax.get_figure().
        """
        # Arrange
        import types
        import matplotlib
        matplotlib.use("Agg")  # backend sin pantalla para tests
        import matplotlib.pyplot as plt
        import matplotlib.axes

        # --- Stub de study de Optuna ---
        dummy_study = object()

        # --- Stub de plot_optimization_history ---
        captured_calls = []

        def fake_plot_optimization_history(study, **kwargs):
            captured_calls.append({"study": study, "kwargs": kwargs})
            fig, ax = plt.subplots()
            return ax

        # --- Harness mínimo que implementa lo que _render_tuning_plots usa ---
        class _PlotHarness:
            logged = []
            t_plot_frame = object()  # sólo se pasa a FigureCanvasTkAgg (mockeado)

            class _FakeTuner:
                @property
                def study(self):
                    return dummy_study

            _tuner = _FakeTuner()

            def t_log(self, text):
                self.logged.append(text)

        harness = _PlotHarness()

        # Monkeypatch del módulo de visualización y del canvas de Tkinter
        import src.mo_module.benchmark.benchmark_gui as gui_module

        fake_canvas_draws = []

        class _FakeCanvas:
            def __init__(self, fig, master):
                fake_canvas_draws.append(fig)

            def draw(self):
                pass

            def get_tk_widget(self):
                class _W:
                    def pack(self, **kw):
                        pass
                return _W()

        monkeypatch.setattr(
            "optuna.visualization.matplotlib.plot_optimization_history",
            fake_plot_optimization_history,
        )
        monkeypatch.setattr(
            "matplotlib.backends.backend_tkagg.FigureCanvasTkAgg",
            _FakeCanvas,
        )

        # Act
        # Ejecutar — no debe lanzar excepción
        BenchmarkGUI._render_tuning_plots(harness)

        # Assert
        # Verificar que no se logueó ningún error
        assert harness.logged == [], f"Se logueó un error: {harness.logged}"

        # Verificar que plot_optimization_history fue llamada SIN 'ax' en kwargs
        assert len(captured_calls) == 1
        assert "ax" not in captured_calls[0]["kwargs"], (
            "plot_optimization_history fue llamada con ax=, lo cual no es válido en Optuna 4.x"
        )

        # Verificar que el canvas recibió una figura matplotlib
        assert len(fake_canvas_draws) == 1
        import matplotlib.figure
        assert isinstance(fake_canvas_draws[0], matplotlib.figure.Figure)
