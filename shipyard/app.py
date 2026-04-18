from __future__ import annotations

from typing import Any

import yaml  # type: ignore[import]
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Footer, Header, Input, Label, Static

from shipyard.services.config_service import (
    load_config,
    load_template_config,
    save_config,
    set_config_path,
    set_template_path,
)
from shipyard.services.polar_service import plot_element_polar, plot_total_polar


class Shipyard(App):
    """Textual TUI for building sailboat configs and plotting polars."""

    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }

    #main {
        width: 98%;
        padding: 1 2;
        background: $boost;
    }

    #title {
        content-align: center middle;
        height: 3;
        color: $accent;
        text-style: bold;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .field-row {
        height: auto;
        margin-bottom: 1;
    }

    .field-label {
        width: 18;
    }

    Label{
    text-align: center;
    }

    Input {
        width: 1fr;
    }

    #content-row {
        height: auto;
        margin-top: 1;
    }

    #yaml_preview {
        height: 1fr;
        padding: 1;
        border: round $accent;
        background: $surface;
        overflow-y: auto;
    }

    #actions Button {
        margin-right: 0;
    }

    #VerticalScroll {
    height: 1fr;
    border: round $accent;
    background: $surface;
    }


    """

    BINDINGS = [
        ("ctrl+s", "save_config", "Save config"),
        ("ctrl+o", "load_config", "Load config"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        yield Vertical(
            Static("Shipyard – Sailboat Config Builder", id="title"),
            Horizontal(
                self._config_panel(),
                Vertical(
                self._preview_panel(),
                ),
                id="content-row",
            ),
            Horizontal(
                Button("Save Config", id="save", variant="success"),
                Button("Load Config", id="load"),
                Button("Plot Keel", id="plot_keel"),
                Button("Plot Sail", id="plot_sail"),
                Button("Plot Rudder", id="plot_rudder"),
                Button("Plot Total Boat", id="plot_total"),
                id="actions",
            ),
            id="main",
        )

        yield Footer()

    def _config_panel(self) -> Vertical:
        return Vertical(
            self._save_load_panel(),
            Collapsible(self._config_keel_panel(), title= "Keel"),
            Collapsible(self._config_sail_panel(), title = "Sail"),
            Collapsible(self._config_rudder_panel(), title = "Rudder"),
            Collapsible(self._config_hull_panel(), title = "Hull"),
        )

    def _config_keel_panel(self) -> Vertical:
        panel = Vertical(
            #Static("Keel", classes="section-title"),
            Horizontal(
            self._field_row("Airfoil", "NACA0012", "keel_airfoil"),
            self._field_row("alpha min/max", "25" ,"keel_alpha_max"),
            self._field_row("Area [m²]", "0.5", "keel_area"),
            ),
            Horizontal(
            self._field_row("x position", "0." ,"keel_x_pos"),
            self._field_row("y position", "0." ,"keel_y_pos"),
            ),
        )
        panel.styles.height = 9
        return panel

    def _config_sail_panel(self) -> Horizontal:
        panel =  Horizontal(
            #Static("Sail", classes="section-title"),
            self._field_row("Airfoil", "NACA0012", "sail_airfoil"),
            self._field_row("Area [m²]", "5.0", "sail_area"),
            #self._field_row("Wind speed [m/s]", "0.0", "sail_wind_speed"),
            #self._field_row("Wind dir [deg]", "0.0", "sail_wind_dir"),
        )
        panel.styles.height = 4
        return panel

    def _config_rudder_panel(self) -> Horizontal:
        panel = Horizontal(
            #Static("Rudder", classes="section-title"),
            self._field_row("Airfoil", "NACA0012", "rudder_airfoil"),
            self._field_row("alpha min/max", "25" ,"rudder_alpha_max"),
            self._field_row("Area [m²]", "0.3", "rudder_area"),
        )
        panel.styles.height = 4
        return panel

    def _config_hull_panel(self) -> Horizontal:
        panel = Horizontal(
            #Static("Hull", classes = "section-title"),
            self._field_row("Xu1", "12.0", "hull_xu1"),
            self._field_row("Yv1", "60.0", "hull_yv1"),
            self._field_row("Nr1", "200.0", "hull_nr1"),
            )
        panel.styles.height = 4
        return panel
    
    def _save_load_panel(self) -> Vertical:
        panel = Vertical(
            Static("Save and Load", classes="section-title"),
            self._field_row("File path (Save)", "boat.yaml", "save_file_path"),
            self._field_row("File path (Load)", "configs/shipyard_template.yaml", "load_file_path"),
        )
        panel.styles.height = 12
        return panel


    @staticmethod
    def _field_row(label: str, default: str, input_id: str) -> Vertical:
        return Vertical(
            Label(label, classes="field-label"),
            Input(default, id=input_id),
            classes="field-row",
        )

    def _preview_panel(self) -> Vertical:
        return Vertical(
            Static("Current YAML (Shipyard template)", classes="section-title"),
            VerticalScroll(Static("", id="yaml_preview")),
        )

    def build_config(self) -> dict[str, dict[str, Any]]:
        """Build a config dictionary aligned with basic_sailbot.yaml."""
        template = load_template_config()

        keel_cfg = {
            "airfoil_name": self.query_one("#keel_airfoil", Input).value,
            "model_type": "keel",
            "alpha_min": -1 * int(self.query_one("#keel_alpha_max", Input).value),
            "alpha_max": int(self.query_one("#keel_alpha_max", Input).value),
            "res": [1e5],
            "area": float(self.query_one("#keel_area", Input).value),
            "x_pos": float(self.query_one("#keel_x_pos", Input).value),
            "y_pos": float(self.query_one("#keel_y_pos", Input).value),
        }

        sail_cfg = {
            "airfoil_name": self.query_one("#sail_airfoil", Input).value,
            "model_type": "sail",
            "alpha_min": -25,
            "alpha_max": 25,
            "res": [1e5],
            "area": float(self.query_one("#sail_area", Input).value),
            "wind_speed": 0,
            "wind_dir_deg": 0,
        }

        rudder_cfg = {
            "airfoil_name": self.query_one("#rudder_airfoil", Input).value,
            "model_type": "rudder",
            "alpha_min": -1 * int(self.query_one("#keel_alpha_max", Input).value),
            "alpha_max": int(self.query_one("#keel_alpha_max", Input).value),
            "res": [1e5],
            "area": float(self.query_one("#rudder_area", Input).value),
        }

        hull_cfg = {
            "Xu1": float(self.query_one("#hull_xu1", Input).value),
            "Yv1": float(self.query_one("#hull_yv1", Input).value),
            "Nr1": float(self.query_one("#hull_nr1", Input).value),
        }

        config: dict[str, dict[str, Any]] = {
            "keel": keel_cfg,
            "sail": sail_cfg,
            "rudder": rudder_cfg,
            "hull" : hull_cfg,
        }

        # Preserve template sections that Shipyard does not currently edit directly.
        for key in ("hull", "quadratic_hull", "boat", "simulation", "environment"):
            section = template.get(key)
            if isinstance(section, dict):
                config[key] = section

        self._update_preview(config)
        return config

    def _update_preview(self, config: dict[str, dict[str, Any]]) -> None:
        preview = yaml.safe_dump(config, sort_keys=False)
        self.query_one("#yaml_preview", Static).update(preview)

    def _load_into_fields(self, config: dict[str, Any]) -> None:
        keel = config.get("keel", {})
        sail = config.get("sail", {})
        rudder = config.get("rudder", {})

        self.query_one("#keel_airfoil", Input).value = str(keel.get("airfoil_name", "NACA0012"))
        self.query_one("#keel_area", Input).value = str(keel.get("area", 0.5))

        self.query_one("#sail_airfoil", Input).value = str(sail.get("airfoil_name", "NACA0012"))
        self.query_one("#sail_area", Input).value = str(sail.get("area", 5.0))
        #self.query_one("#sail_wind_speed", Input).value = str(sail.get("wind_speed", 0.0))
        #self.query_one("#sail_wind_dir", Input).value = str(sail.get("wind_dir_deg", 0.0))

        self.query_one("#rudder_airfoil", Input).value = str(rudder.get("airfoil_name", "NACA0012"))
        self.query_one("#rudder_area", Input).value = str(rudder.get("area", 0.3))

        # Also refresh the preview using the full config.
        self._update_preview(config)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "load":
            config = load_config()
            if config:
                self._load_into_fields(config)
            return

        config = self.build_config()

        if button_id == "save":
            save_config(config)
        elif button_id == "plot_keel":
            plot_element_polar(config["keel"])
        elif button_id == "plot_sail":
            plot_element_polar(config["sail"])
        elif button_id == "plot_rudder":
            plot_element_polar(config["rudder"])
        elif button_id == "plot_total":
            plot_total_polar(config)

    def action_save_config(self) -> None:
        set_config_path(self.query_one("#save_file_path", Input).value)
        save_config(self.build_config())

    def action_load_config(self) -> None:
        set_template_path(self.query_one("#load_file_path", Input).value)
        config = load_config()
        if config:
            self._load_into_fields(config)


if __name__ == "__main__":
    Shipyard().run()
