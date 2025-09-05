"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

- This module provides utility functions for visualizing decision trees as PNG images.
- It estimates the required image dimensions based on the tree's text representation,
and safely saves the visualization to a PNG file using matplotlib. The module ensures
that the generated image does not exceed specified dimension limits, and allows for
customization of font size, DPI, and padding. If the tree is too large to render,
the module prints a warning and does not generate the file.
"""

from typing import Tuple
import matplotlib.pyplot as plt

# Default configuration values for rendering the tree visualization
_DEFAULTS = {
    "max_dim_px": 64000,   # Maximum allowed dimension (width or height) in pixels
    "font_size": 12,       # Default font size in points
    "dpi": 200,            # Default dots per inch for the image
    "padding_px": 24,      # Default padding around the tree text in pixels
}

def _estimate_tree_dimensions(tree_text: str, font_size: int, dpi: int, padding_px: int) -> Tuple[int, int]:
    """
    Estimate the required width and height in pixels to render the tree text.

    Parameters:
        tree_text (str): The string representation of the tree.
        font_size (int): Font size in points.
        dpi (int): Dots per inch for the image.
        padding_px (int): Padding in pixels around the tree text.

    Returns:
        Tuple[int, int]: Estimated (width_px, height_px) in pixels.
    """
    # Split the tree text into lines for dimension calculation
    lines = tree_text.splitlines() if tree_text else [""]
    # Calculate pixels per point for the given DPI
    pixels_per_point = dpi / 72.0
    # Estimate average character width in pixels (monospace font)
    char_width_px = font_size * 0.6 * pixels_per_point
    # Estimate line height in pixels, ensuring a minimum of 1.0
    line_height_px = max(1.0, font_size * 1.4 * pixels_per_point)
    # Find the maximum number of characters in any line
    max_characters = max((len(line) for line in lines), default=1)
    # Calculate total width including padding
    width_px = int(padding_px * 2 + max(1, max_characters) * char_width_px)
    # Calculate total height including padding
    height_px = int(padding_px * 2 + max(1, len(lines)) * line_height_px)
    return width_px, height_px

def save_tree_png_safe(
    tree_text: str,
    filename: str,
    max_dim_px: int = None,
    font_size: int = None,
    dpi: int = None,
    padding_px: int = None
):
    """
    Render the tree text as a PNG image and save it to the specified file.
    If the estimated image dimensions exceed the allowed maximum, the function
    prints a warning and does not generate the file.

    Parameters:
        tree_text (str): The string representation of the tree to visualize.
        filename (str): The path to save the PNG image.
        max_dim_px (int, optional): Maximum allowed dimension (width or height) in pixels.
                                    If None, uses the default value.
        font_size (int, optional): Font size in points for rendering the tree text.
                                   If None, uses the default value.
        dpi (int, optional): Dots per inch for the image. If None, uses the default value.
        padding_px (int, optional): Padding in pixels around the tree text.
                                    If None, uses the default value.

    Returns:
        None
    """
    # Use default values for parameters if not provided
    max_dim_px = _DEFAULTS["max_dim_px"] if max_dim_px is None else max_dim_px
    font_size = _DEFAULTS["font_size"] if font_size is None else font_size
    dpi = _DEFAULTS["dpi"] if dpi is None else dpi
    padding_px = _DEFAULTS["padding_px"] if padding_px is None else padding_px

    # Estimate the required image dimensions for the tree text
    width_px, height_px = _estimate_tree_dimensions(tree_text, font_size, dpi, padding_px)

    # Check if the estimated dimensions exceed the allowed maximum
    if width_px > max_dim_px or height_px > max_dim_px:
        print("Cannot generate tree PNG: dimensions exceed the allowed maximum.")
        return

    try:
        # Calculate figure size in inches for matplotlib
        figure_width_in = max(2.0, width_px / dpi)
        figure_height_in = max(2.0, height_px / dpi)

        # Create a matplotlib figure with the calculated size and DPI
        fig = plt.figure(figsize=(figure_width_in, figure_height_in), dpi=dpi)
        # Add an axes that fills the entire figure (no margins)
        ax = fig.add_axes([0, 0, 1, 1])
        # Hide axes for a clean visualization
        ax.axis("off")

        # Render the tree text at the top-left corner using monospace font
        ax.text(
            0.01, 0.99, tree_text,
            fontfamily="monospace",
            fontsize=font_size,
            va="top",
            ha="left"
        )

        # Save the figure to the specified filename with tight bounding box and small padding
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
        # Close the figure to free resources
        plt.close(fig)
    except Exception:
        # Handle any errors during rendering or saving
        print("Cannot generate tree PNG: an error occurred during rendering or saving.")