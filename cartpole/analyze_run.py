"""Analyze and visualize cartpole simulation results.

Loads the most recent log file from cartpole/logs/ and creates interactive plots
showing system performance over time.
"""

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
import numpy as np

from tuning.metrics import settling_time, overshoot


def find_most_recent_log(log_dir: Path) -> Path:
    """Find the most recent CSV log file in the logs directory."""
    log_files = list(log_dir.glob("state_log_*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    # Sort by modification time, most recent first
    most_recent = max(log_files, key=lambda p: p.stat().st_mtime)
    return most_recent


def load_log_data(log_file: Path) -> pl.DataFrame:
    """Load CSV log data into a Polars DataFrame."""
    df = pl.read_csv(log_file)

    # Add derived metrics using Polars expressions
    df = df.with_columns([
        (pl.col('step') * (1.0/60.0)).alias('time'),  # Assuming 60 Hz physics
        (pl.col('pole_angle') * 180 / np.pi).alias('pole_angle_deg'),  # Convert to degrees
        pl.col('pole_angle').abs().alias('abs_pole_angle'),
        pl.col('cart_position').abs().alias('abs_cart_position')
    ])

    return df


def create_plots(df: pl.DataFrame, output_path: Path = None):
    """Create interactive visualization of cartpole performance metrics.

    Creates a multi-panel figure showing:
    - Pole angle and angular velocity
    - Cart position and velocity
    - Control force
    - Phase portraits
    """
    # Color palette (following dataviz guidelines - sequential for magnitude)
    colors = {
        'angle': '#2563eb',      # Blue for angle
        'velocity': '#dc2626',   # Red for velocity
        'position': '#059669',   # Green for position
        'force': '#7c3aed'       # Purple for force
    }

    # Create subplots: 3 rows x 3 columns
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            'Pole Angle Error', 'Pole Angular Velocity', 'Pole Phase Portrait',
            'Cart Position Error', 'Cart Velocity', 'Cart Phase Portrait',
            'LQR Control Force', 'Error Magnitude (Log Scale)', 'Performance Summary'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'table'}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.08
    )

    # Convert to numpy for plotly (or use .to_list() for native Python lists)
    time = df['time'].to_numpy()
    pole_angle_deg = df['pole_angle_deg'].to_numpy()
    pole_angular_vel = df['pole_angular_vel'].to_numpy()
    cart_position = df['cart_position'].to_numpy()
    cart_velocity = df['cart_velocity'].to_numpy()
    control_force = df['control_force'].to_numpy()
    abs_pole_angle = df['abs_pole_angle'].to_numpy()
    abs_cart_position = df['abs_cart_position'].to_numpy()

    # ========== Row 1: Pole Dynamics ==========
    # Pole angle over time
    fig.add_trace(
        go.Scatter(x=time, y=pole_angle_deg, mode='lines',
                   name='Pole Angle', line=dict(color=colors['angle'], width=2),
                   hovertemplate='Time: %{x:.3f}s<br>Angle: %{y:.3f}°<extra></extra>'),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # Pole angular velocity over time
    fig.add_trace(
        go.Scatter(x=time, y=pole_angular_vel, mode='lines',
                   name='Angular Velocity', line=dict(color=colors['velocity'], width=2),
                   hovertemplate='Time: %{x:.3f}s<br>Velocity: %{y:.4f} rad/s<extra></extra>'),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)

    # Phase portrait: Pole
    fig.add_trace(
        go.Scatter(x=pole_angle_deg, y=pole_angular_vel, mode='markers',
                   name='Pole Phase',
                   marker=dict(size=6, color=time, colorscale='Viridis',
                              showscale=True, colorbar=dict(x=1.0, len=0.3, y=0.83)),
                   hovertemplate='Angle: %{x:.3f}°<br>Velocity: %{y:.4f} rad/s<extra></extra>'),
        row=1, col=3
    )

    # ========== Row 2: Cart Dynamics ==========
    # Cart position over time
    fig.add_trace(
        go.Scatter(x=time, y=cart_position, mode='lines',
                   name='Cart Position', line=dict(color=colors['position'], width=2),
                   hovertemplate='Time: %{x:.3f}s<br>Position: %{y:.4f} m<extra></extra>'),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    # Cart velocity over time
    fig.add_trace(
        go.Scatter(x=time, y=cart_velocity, mode='lines',
                   name='Cart Velocity', line=dict(color=colors['velocity'], width=2),
                   hovertemplate='Time: %{x:.3f}s<br>Velocity: %{y:.4f} m/s<extra></extra>'),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=2)

    # Phase portrait: Cart
    fig.add_trace(
        go.Scatter(x=cart_position, y=cart_velocity, mode='markers',
                   name='Cart Phase',
                   marker=dict(size=6, color=time, colorscale='Viridis',
                              showscale=True, colorbar=dict(x=1.0, len=0.3, y=0.5)),
                   hovertemplate='Position: %{x:.4f} m<br>Velocity: %{y:.4f} m/s<extra></extra>'),
        row=2, col=3
    )

    # ========== Row 3: Control and Performance ==========
    # Control force over time
    fig.add_trace(
        go.Scatter(x=time, y=control_force, mode='lines',
                   name='Control Force', line=dict(color=colors['force'], width=2),
                   hovertemplate='Time: %{x:.3f}s<br>Force: %{y:.2f} N<extra></extra>'),
        row=3, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    # Error metrics over time (log scale)
    fig.add_trace(
        go.Scatter(x=time, y=abs_pole_angle, mode='lines',
                   name='|Pole Angle|', line=dict(color=colors['angle'], width=2),
                   hovertemplate='Time: %{x:.3f}s<br>|Angle|: %{y:.4f} rad<extra></extra>'),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=time, y=abs_cart_position, mode='lines',
                   name='|Cart Position|', line=dict(color=colors['position'], width=2),
                   hovertemplate='Time: %{x:.3f}s<br>|Position|: %{y:.4f} m<extra></extra>'),
        row=3, col=2
    )

    # Summary statistics table (using Polars aggregations)
    final_pole_angle = abs(pole_angle_deg[-1])
    max_pole_angle = np.degrees(df['abs_pole_angle'].max())
    final_cart_pos = abs(cart_position[-1])
    max_cart_pos = df['abs_cart_position'].max()
    mean_force = df.select(pl.col('control_force').abs().mean()).item()
    max_force = df.select(pl.col('control_force').abs().max()).item()
    st = settling_time(time, df['pole_angle'].to_numpy())
    ov = overshoot(df['pole_angle'].to_numpy(), df['pole_angle'][0])

    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'],
                       fill_color='lightgray',
                       align='left',
                       font=dict(size=12, color='black')),
            cells=dict(values=[
                ['Final Pole Angle', 'Max Pole Angle', 'Final Cart Position',
                 'Max Cart Position', 'Mean |Force|', 'Max |Force|',
                 'Settling Time', 'Overshoot', 'Simulation Time', 'Steps'],
                [f'{final_pole_angle:.3f}°', f'{max_pole_angle:.3f}°',
                 f'{final_cart_pos:.4f} m', f'{max_cart_pos:.4f} m',
                 f'{mean_force:.2f} N', f'{max_force:.2f} N',
                 f'{st:.3f} s' if np.isfinite(st) else 'never',
                 f'{ov:.1%}',
                 f'{time[-1]:.2f} s', f'{len(df)}']
            ],
            fill_color='white',
            align='left',
            font=dict(size=11))
        ),
        row=3, col=3
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Angle (°)", row=1, col=1)

    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=1, col=2)

    fig.update_xaxes(title_text="Angle (°)", row=1, col=3)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=1, col=3)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Position (m)", row=2, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=2)

    fig.update_xaxes(title_text="Position (m)", row=2, col=3)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=3)

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Force (N)", row=3, col=1)

    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Absolute Error", type="log", row=3, col=2)

    # Update layout
    fig.update_layout(
        height=1000,
        width=1600,
        title_text="Cartpole LQR Control Performance Analysis",
        title_font_size=20,
        showlegend=False,
        hovermode='closest',
        template='plotly_white'
    )

    # Save as HTML for interactivity
    if output_path:
        html_path = output_path.with_suffix('.html')
        fig.write_html(html_path)
        print(f"Interactive plot saved to: {html_path}")
    else:
        fig.show()

    return fig


def main():
    """Main analysis workflow."""
    # Find most recent log
    log_dir = Path("cartpole/logs")

    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist.")
        return

    log_file = find_most_recent_log(log_dir)
    print(f"Analyzing: {log_file.name}")
    from datetime import datetime
    print(f"Modified: {datetime.fromtimestamp(log_file.stat().st_mtime)}")
    print()

    # Load data
    df = load_log_data(log_file)
    final_time = df['time'][-1]
    print(f"Loaded {len(df)} steps ({final_time:.2f} seconds of simulation)")
    print()

    # Create plots
    output_path = log_dir / f"{log_file.stem}_analysis"
    create_plots(df, output_path)

    # Print quick summary (using Polars expressions)
    print("\n=== Quick Summary ===")
    print(f"Final pole angle: {df['pole_angle_deg'][-1]:.3f}°")
    print(f"Final cart position: {df['cart_position'][-1]:.4f} m")
    print(f"Mean control force: {df.select(pl.col('control_force').abs().mean()).item():.2f} N")
    print(f"Max control force: {df.select(pl.col('control_force').abs().max()).item():.2f} N")


if __name__ == "__main__":
    main()
