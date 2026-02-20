import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def confidence_interval_for_proportion(n, p, N, confidence=0.95):
    """
    Calculate the confidence interval for an estimated population proportion.
    """
    # Get the z-value for the desired confidence level
    z = norm.ppf((1 + confidence) / 2)
    
    # Calculate the finite population correction factor
    fpc = np.sqrt((N - n) / (N - 1))
    
    # Calculate standard error with finite population correction
    se = np.sqrt((p * (1 - p)) / n) * fpc
    
    # Margin of error
    margin_of_error = z * se
    
    # Confidence interval bounds
    ci_lower = max(0, p - margin_of_error)
    ci_upper = min(1, p + margin_of_error)
    
    return {
        'margin_of_error': margin_of_error,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'sample_estimate': p
    }


def find_sample_sizes_for_precision(target_me, N, min_sample_size, max_sample_size, num_sample_steps,
                                     min_proportion, max_proportion, num_proportion_steps):
    """
    Find sample size combinations that achieve a target margin of error.
    """
    
    # Generate ranges based on number of steps
    sample_sizes = np.linspace(min_sample_size, max_sample_size, num_sample_steps, dtype=int)
    proportions = np.linspace(min_proportion, max_proportion, num_proportion_steps)
    
    # Grid search: calculate margin of error for all combinations
    results = []
    me_matrix = np.zeros((len(proportions), len(sample_sizes)))
    
    for i, p in enumerate(proportions):
        for j, n in enumerate(sample_sizes):
            result = confidence_interval_for_proportion(n, p, N)
            me = result['margin_of_error']
            me_matrix[i, j] = me
            
            results.append({
                'sample_size': n,
                'population_proportion': p,
                'margin_of_error': me
            })
    
    df_results = pd.DataFrame(results)
    
    # Find combinations close to target (within ±10% of target)
    tolerance = target_me * 0.10
    close_combos = df_results[
        (df_results['margin_of_error'] >= target_me - tolerance) &
        (df_results['margin_of_error'] <= target_me + tolerance)
    ].copy()
    
    # Sort by how close they are to target
    close_combos['distance_from_target'] = abs(close_combos['margin_of_error'] - target_me)
    close_combos = close_combos.sort_values('distance_from_target')
    
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use a soft color palette (YlOrRd = Yellow-Orange-Red)
    im = ax.imshow(me_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    
    # Set axis labels
    ax.set_xticks(np.arange(len(sample_sizes)))
    ax.set_yticks(np.arange(len(proportions)))
    ax.set_xticklabels([f'{int(n)}' for n in sample_sizes], rotation=45, ha='right')
    ax.set_yticklabels([f'{p:.2f}' for p in proportions])
    
    ax.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Population Proportion', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Margin of Error Grid Search\n(Target ME = ±{target_me*100:.1f}%, Population = {N:,})',
        fontsize=13, fontweight='bold',
        pad=20
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Margin of Error')
    
    # Add text annotations for a subset (to avoid clutter)
    sample_step = max(1, len(sample_sizes) // 8)
    prop_step = max(1, len(proportions) // 8)
    
    for i in range(0, len(proportions), prop_step):
        for j in range(0, len(sample_sizes), sample_step):
            text = ax.text(j, i, f'{me_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    return fig, close_combos, df_results


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Power & Precision Analysis", layout="wide")

st.title("📊 Survey Sample Size & Precision Analysis")
st.markdown("""
This tool helps you find the right sample size for your survey by analyzing how 
**sample size** and **population proportion** affect the **margin of error**.
""")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Survey Parameters")
    
    target_me = st.slider(
        "Target Margin of Error (±)",
        min_value=0.01,
        max_value=0.50,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="How precise do you need your estimate to be? (e.g., 0.05 = ±5%)"
    )
    
    population_size = st.number_input(
        "Population Size (N)",
        min_value=100,
        value=10000,
        step=100,
        help="Total population you're surveying from"
    )
    
    st.markdown("---")
    st.subheader("🔍 Search Grid")
    
    col_min_n, col_max_n = st.columns(2)
    with col_min_n:
        min_sample = st.number_input(
            "Min Sample Size",
            min_value=10,
            value=50,
            step=10,
            help="Smallest sample size to test"
        )
    
    with col_max_n:
        max_sample = st.number_input(
            "Max Sample Size",
            min_value=100,
            value=2000,
            step=100,
            help="Largest sample size to test"
        )
    
    num_sample_steps = st.slider(
        "Number of Sample Size Steps",
        min_value=3,
        max_value=30,
        value=15,
        help="How many sample sizes to test between min and max"
    )
    
    st.markdown("---")
    
    col_min_p, col_max_p = st.columns(2)
    with col_min_p:
        min_prop = st.slider(
            "Min Proportion",
            min_value=0.01,
            max_value=0.49,
            value=0.05,
            step=0.01,
            help="Smallest population proportion to test"
        )
    
    with col_max_p:
        max_prop = st.slider(
            "Max Proportion",
            min_value=min_prop + 0.01,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Largest population proportion to test"
        )
    
    num_prop_steps = st.slider(
        "Number of Proportion Steps",
        min_value=2,
        max_value=20,
        value=10,
        help="How many proportions to test between min and max"
    )
    
    run_button = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

# Right column for results
with col2:
    if run_button:
        with st.spinner("Running analysis..."):
            fig, matching_combos, all_results = find_sample_sizes_for_precision(
                target_me=target_me,
                N=int(population_size),
                min_sample_size=int(min_sample),
                max_sample_size=int(max_sample),
                num_sample_steps=int(num_sample_steps),
                min_proportion=min_prop,
                max_proportion=max_prop,
                num_proportion_steps=int(num_prop_steps)
            )
        
        # Display the heatmap
        st.subheader("🎨 Margin of Error Heatmap")
        st.pyplot(fig, use_container_width=True)
        
        # Display results summary
        st.markdown("---")
        st.subheader("📈 Results Summary")
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric(
                "Target ME",
                f"±{target_me*100:.2f}%"
            )
        
        with col_summary2:
            st.metric(
                "Matches Found",
                len(matching_combos),
                delta=f"within ±{target_me*10:.1f}% tolerance"
            )
        
        with col_summary3:
            st.metric(
                "Population Size",
                f"{int(population_size):,}"
            )
        
        # Display the matching combinations table
        if len(matching_combos) > 0:
            st.markdown("---")
            st.subheader("✅ Combinations Achieving Target Precision")
            
            # Format the table for display
            display_df = matching_combos.head(20)[['sample_size', 'population_proportion', 'margin_of_error']].copy()
            display_df['sample_size'] = display_df['sample_size'].astype(int)
            display_df['population_proportion'] = display_df['population_proportion'].apply(lambda x: f'{x:.3f}')
            display_df['margin_of_error'] = display_df['margin_of_error'].apply(lambda x: f'±{x*100:.3f}%')
            display_df.columns = ['Sample Size', 'Population Proportion', 'Achieved ME']
            
            st.dataframe(display_df, use_container_width=True)
            
            st.caption(f"Showing top 20 of {len(matching_combos)} matches")
            
            # Download button
            csv_data = matching_combos[['sample_size', 'population_proportion', 'margin_of_error']].copy()
            csv_data = csv_data.head(100)
            csv_string = csv_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv_string,
                file_name="precision_analysis_results.csv",
                mime="text/csv"
            )
        
        else:
            st.warning(
                f"❌ No combinations found within tolerance (±{target_me*10:.1f}%).\n\n"
                f"Try adjusting:\n"
                f"- Increasing max sample size\n"
                f"- Relaxing target margin of error\n"
                f"- Expanding the search grid"
            )
            
            # Show closest matches
            st.markdown("---")
            st.subheader("🎯 Closest Matches")
            all_results['distance_from_target'] = abs(all_results['margin_of_error'] - target_me)
            closest = all_results.nsmallest(10, 'distance_from_target')
            
            display_closest = closest[['sample_size', 'population_proportion', 'margin_of_error']].copy()
            display_closest['sample_size'] = display_closest['sample_size'].astype(int)
            display_closest['population_proportion'] = display_closest['population_proportion'].apply(lambda x: f'{x:.3f}')
            display_closest['margin_of_error'] = display_closest['margin_of_error'].apply(lambda x: f'±{x*100:.3f}%')
            display_closest.columns = ['Sample Size', 'Population Proportion', 'Achieved ME']
            
            st.dataframe(display_closest, use_container_width=True)
    
    else:
        st.info("👈 Configure your parameters and click 'Run Analysis' to get started!")
        
        st.markdown("""
        ### How to Use This Tool
        
        1. **Set your target precision**: How precise should your estimate be?
        2. **Enter population size**: How many people are you surveying?
        3. **Define search ranges**: What sample sizes and proportions should we test?
        4. **Run the analysis**: Click the button to see results!
        
        ### Understanding the Results
        
        - **Heatmap**: Shows margin of error across all combinations (yellow = lower error, red = higher error)
        - **Table**: Lists sample size + proportion combinations that achieve your target precision
        - **Tolerance**: Results within ±10% of your target are highlighted
        """)
