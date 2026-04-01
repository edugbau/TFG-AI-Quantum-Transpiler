"""Synthesis-specific controls for the RL GUI."""

import customtkinter as ctk


class SynthesisView(ctk.CTkFrame):
    """Sidebar subview that groups synthesis-only controls."""

    def __init__(self, parent, *, basis_profiles):
        super().__init__(parent, fg_color="transparent")

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Synthesis Basis:", anchor="w").grid(
            row=0, column=0, padx=0, pady=(10, 0), sticky="w"
        )
        self.basis_profile_option = ctk.CTkOptionMenu(self, values=list(basis_profiles))
        self.basis_profile_option.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")

    def show(self):
        self.grid()
        for child in self.winfo_children():
            child.grid()

    def hide(self):
        for child in self.winfo_children():
            child.grid_remove()
        self.grid_remove()
