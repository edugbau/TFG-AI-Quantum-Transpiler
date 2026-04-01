"""Routing-specific controls for the RL GUI."""

import customtkinter as ctk


class RoutingView(ctk.CTkFrame):
    """Sidebar subview that groups routing-only controls."""

    def __init__(self, parent, *, on_lookahead_change):
        super().__init__(parent, fg_color="transparent")

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Frontier:", anchor="w").grid(
            row=0, column=0, padx=0, pady=(10, 0), sticky="w"
        )
        self.frontier_option = ctk.CTkOptionMenu(self, values=["sequential", "dag"])
        self.frontier_option.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")

        self.lookahead_label = ctk.CTkLabel(self, text="Lookahead: 10", anchor="w")
        self.lookahead_label.grid(row=2, column=0, padx=0, pady=(10, 0), sticky="w")
        self.lookahead_slider = ctk.CTkSlider(
            self,
            from_=3,
            to=20,
            number_of_steps=17,
            command=on_lookahead_change,
        )
        self.lookahead_slider.grid(row=3, column=0, padx=0, pady=(0, 10), sticky="ew")

    def show(self):
        self.grid()
        for child in self.winfo_children():
            child.grid()

    def hide(self):
        for child in self.winfo_children():
            child.grid_remove()
        self.grid_remove()
