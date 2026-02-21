from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Footer, Header, Input, Static

from shipyard.services.polar_service import generate_and_plot


class Shipyard(App):
    """Main application class for shipyard."""

    CSS = """
    Screen {
        align: center middle;
    }

    Vertical {
        width: 50%;
    }

    Input {
        margin-bottom: 1;
    }

    Button {
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Static("Sailbench Studio - Polar Generator"),
            Input(value="NACA0012", id="airfoil"),
            Input(value="5e5", id="re"),
            Button("Generate Polar", id="generate"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "generate":
            airfoil = self.query_one("#airfoil", Input).value
            re_value = float(self.query_one("#re", Input).value)

            generate_and_plot(airfoil, re_value)


if __name__ == "__main__":
    Shipyard().run()
