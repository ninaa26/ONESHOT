from __future__ import annotations

from typing import Any

import yaml  # type: ignore[import]
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, Static

from shipyard.services.config_service import load_config, load_template_config, save_config
from shipyard.services.polar_service import plot_element_polar, plot_total_polar


class Shipyard(App):
    """Textual TUI for building sailboat configs and plotting polars."""

    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }

    #main {
        width: 90%;
        border: heavy $accent;
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

    Input {
        width: 18;
    }

    #content-row {
        height: auto;
        margin-top: 1;
    }

    #yaml_preview {
        height: 16;
        padding: 1;
        border: round $accent;
        background: $surface;
    }

    #actions Button {
        margin-right: 2;
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
                self._preview_panel(),
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
            Static("Keel", classes="section-title"),
            self._field_row("Airfoil", "NACA0012", "keel_airfoil"),
            self._field_row("Area [m²]", "0.5", "keel_area"),
            Static("Sail", classes="section-title"),
            self._field_row("Airfoil", "NACA0012", "sail_airfoil"),
            self._field_row("Area [m²]", "5.0", "sail_area"),
            self._field_row("Wind speed [m/s]", "0.0", "sail_wind_speed"),
            self._field_row("Wind dir [deg]", "0.0", "sail_wind_dir"),
            Static("Rudder", classes="section-title"),
            self._field_row("Airfoil", "NACA0012", "rudder_airfoil"),
            self._field_row("Area [m²]", "0.3", "rudder_area"),
        )

    @staticmethod
    def _field_row(label: str, default: str, input_id: str) -> Horizontal:
        return Horizontal(
            Label(label, classes="field-label"),
            Input(default, id=input_id),
            classes="field-row",
        )

    def _preview_panel(self) -> Vertical:
        return Vertical(
            Static("Current YAML (Shipyard template)", classes="section-title"),
            Static("", id="yaml_preview"),
        )

    def build_config(self) -> dict[str, dict[str, Any]]:
        """Build a config dictionary aligned with basic_sailbot.yaml."""
        template = load_template_config()

        keel_cfg = {
            "airfoil_name": self.query_one("#keel_airfoil", Input).value,
            "model_type": "keel",
            "alpha_min": -25,
            "alpha_max": 25,
            "res": [1e5],
            "area": float(self.query_one("#keel_area", Input).value),
            "x_pos": template.get("keel", {}).get("x_pos", 0.0),
            "y_pos": template.get("keel", {}).get("y_pos", 0.0),
        }

        sail_cfg = {
            "airfoil_name": self.query_one("#sail_airfoil", Input).value,
            "model_type": "sail",
            "alpha_min": -25,
            "alpha_max": 25,
            "res": [1e5],
            "area": float(self.query_one("#sail_area", Input).value),
            "wind_speed": float(self.query_one("#sail_wind_speed", Input).value),
            "wind_dir_deg": float(self.query_one("#sail_wind_dir", Input).value),
        }

        rudder_cfg = {
            "airfoil_name": self.query_one("#rudder_airfoil", Input).value,
            "model_type": "rudder",
            "alpha_min": -25,
            "alpha_max": 25,
            "res": [1e5],
            "area": float(self.query_one("#rudder_area", Input).value),
        }

        config: dict[str, dict[str, Any]] = {
            "keel": keel_cfg,
            "sail": sail_cfg,
            "rudder": rudder_cfg,
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
        self.query_one("#sail_wind_speed", Input).value = str(sail.get("wind_speed", 0.0))
        self.query_one("#sail_wind_dir", Input).value = str(sail.get("wind_dir_deg", 0.0))

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
        save_config(self.build_config())

    def action_load_config(self) -> None:
        config = load_config()
        if config:
            self._load_into_fields(config)


if __name__ == "__main__":
    Shipyard().run()
