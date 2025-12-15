import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Comparison of Jax, numpy and PyTorch on Apple's silicon

    This notebook shows the results of the trace calculation of a product of matrices
    $$R = A_{ij}B_{jk}A_{kl}B_{lm}A_{mn}B_{ni} = \text{tr}\left(ABABAB\right)$$

    For each librairy we compare
    * The Einstein summation
    * With the matmul and trace operators
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pickle
    import plotly.graph_objects as go

    # --- Cell 1: Configuration & Colors ---
    # We define colors and mapping logic here to keep the plotting cell clean.
    colors = {
        "numpy": "rgb(31, 119, 180)",  # tab:blue approximation
        "jax_matmul": "rebeccapurple",
        "jax_einsum": "mediumorchid",
        "torch_mat_cpu": "#b22222",    # Firebrick
        "torch_ein_cpu": "#f08080",    # Light Coral
        "torch_mat_mps": "#ff8c00",    # Dark Orange
        "torch_ein_mps": "#ffbd0a",    # Saffron/Gold
    }

    # Map user-friendly labels to the internal data keys
    # Structure: (Label): (Library Key, Device Suffix for Torch)
    options_map = {
        "NumPy": ["numpy"],
        "JAX (CPU)": ["jax"],
        "PyTorch (CPU)": ["torch_cpu"],
        "PyTorch (MPS)": ["torch_mps"]
    }


    # --- Cell 2: Data Loading ---
    # We use a try-except block to prevent the app from crashing if files are missing.
    # Replace the empty dicts with your actual data if files are loaded successfully.
    results_m4 = {}
    results_m2 = {}

    try:
        with open("results_jax_torch_2025-12-15.pkl", "rb") as file:
            results_m4 = pickle.load(file)
        with open("results_jax_torch_ssh_2025-12-15.pkl", "rb") as file:
            results_m2 = pickle.load(file)
        status_msg = mo.md("**Data loaded successfully.**")
    except FileNotFoundError as e:
        status_msg = mo.md(f"**Warning:** Could not find data files ({e}). Please ensure .pkl files are in the directory.")


    # --- Cell 3: UI Controls ---
    # Create controls for filtering
    filter_selector = mo.ui.multiselect(
        options=list(options_map.keys()),
        value=list(options_map.keys()), # Select all by default
        label="**Select Library & Device:**"
    )

    dataset_selector = mo.ui.multiselect(
        options=["M4 Pro (Local)", "M2 Ultra (SSH)"],
        value=["M4 Pro (Local)", "M2 Ultra (SSH)"], # Select all by default
        label="**Select Chips:**"
    )

    # Display the controls
    mo.vstack([
        status_msg,
        mo.hstack([filter_selector, dataset_selector], justify="start")
    ])

    return (
        colors,
        dataset_selector,
        filter_selector,
        go,
        mo,
        results_m2,
        results_m4,
    )


@app.cell
def _():
    return


@app.cell
def _(
    colors,
    dataset_selector,
    filter_selector,
    go,
    mo,
    results_m2,
    results_m4,
):



    # --- Cell 4: Plotting Logic ---
    # This cell reacts whenever the selectors in Cell 3 change.
    def create_plot(selected_filters, selected_datasets):
        fig = go.Figure()

        # Helper to add traces dynamically
        def add_trace(data_source, dataset_label, line_mode):
            if not data_source: return 
        
            # NumPy
            if "NumPy" in selected_filters and "numpy" in data_source:
                x_data = list(dict.fromkeys(data_source["numpy"]["N"]))
                fig.add_trace(go.Scatter(
                    x=x_data, y=data_source["numpy"]["matmul"],
                    mode=line_mode, name=f"{dataset_label} np matmul",
                    line=dict(color=colors["numpy"], width=2),
                    legendgroup="numpy"
                ))

            # JAX
            if "JAX (CPU)" in selected_filters and "jax" in data_source:
                x_data = list(dict.fromkeys(data_source["jax"]["N"]))
                fig.add_trace(go.Scatter(
                    x=x_data, y=data_source["jax"]["matmul"],
                    mode=line_mode, name=f"{dataset_label} jax matmul",
                    line=dict(color=colors["jax_matmul"], width=2),
                    legendgroup="jax"
                ))
                fig.add_trace(go.Scatter(
                    x=x_data, y=data_source["jax"]["einsum"],
                    mode=line_mode, name=f"{dataset_label} jax einsum",
                    line=dict(color=colors["jax_einsum"]),
                    legendgroup="jax"
                ))

            # Torch
            if "torch" in data_source:
                x_data = list(dict.fromkeys(data_source["torch"]["N"]))
            
                # Torch CPU
                if "PyTorch (CPU)" in selected_filters:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=data_source["torch"]["matmul_cpu"],
                        mode=line_mode, name=f"{dataset_label} torch matmul_cpu",
                        line=dict(color=colors["torch_mat_cpu"], width=2),
                        legendgroup="torch_cpu"
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_data, y=data_source["torch"]["einsum_cpu"],
                        mode=line_mode, name=f"{dataset_label} torch einsum_cpu",
                        line=dict(color=colors["torch_ein_cpu"]),
                        legendgroup="torch_cpu"
                    ))
                
                # Torch MPS
                if "PyTorch (MPS)" in selected_filters:
                    fig.add_trace(go.Scatter(
                        x=x_data, y=data_source["torch"]["matmul_mps"],
                        mode=line_mode, name=f"{dataset_label} torch matmul_mps",
                        line=dict(color=colors["torch_mat_mps"], width=2),
                        legendgroup="torch_mps"
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_data, y=data_source["torch"]["einsum_mps"],
                        mode=line_mode, name=f"{dataset_label} torch einsum_mps",
                        line=dict(color=colors["torch_ein_mps"]),
                        legendgroup="torch_mps"
                    ))

        # Add traces based on dataset selection
        # M4 Pro uses Lines only
        if "M4 Pro (Local)" in selected_datasets:
            add_trace(results_m4, "M4 Pro", "lines")
    
        # M2 Ultra uses Lines + Markers (to mimic the "-+" style)
        if "M2 Ultra (SSH)" in selected_datasets:
            add_trace(results_m2, "M2 Ultra", "lines+markers")

        # Layout Updates
        fig.update_layout(
            title="Matrix Multiplication Performance (Log Scale)",
            xaxis_title="Matrix Size (N)",
            yaxis_title="Duration (ms)",
            yaxis_type="log",
            height=600,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                groupclick="toggleitem" # Click legend group to toggle all traces in that group
            )
        )
        fig.update_yaxes(exponentformat = 'power')
        return fig

    # Render the plot with current selection
    mo.ui.plotly(create_plot(filter_selector.value, dataset_selector.value))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
