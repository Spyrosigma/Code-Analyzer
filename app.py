import streamlit as st
import tempfile
import os
import git
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from pathlib import Path
import shutil
import time

from main import analyze_codebase

# [Previous CodeAnalyzer class and related code would go here]

def clone_github_repo(repo_url: str) -> str:
    """Clone a GitHub repository to a temporary directory."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Clone the repository
        git.Repo.clone_from(repo_url, temp_dir)
        return temp_dir
    except git.exc.GitCommandError as e:
        st.error(f"Error cloning repository: {e}")
        return None

def create_dependency_network(report):
    """Create a network graph of function dependencies."""
    G = nx.DiGraph()
    
    # Add nodes and edges from function relationships
    for file, funcs in report['function_relationships'].items():
        for func, calls in funcs.items():
            G.add_node(func)
            for called_func in calls:
                G.add_edge(func, called_func)
    
    return G

def plot_dependency_network(G):
    """Create a Plotly figure for the dependency network."""
    pos = nx.spring_layout(G)
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create nodes
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(node) for node in G.nodes()],
        textposition="bottom center",
        marker=dict(
            size=10,
            color='#007bff',
            line_width=2)))
    
    fig.update_layout(
        title='Code Dependency Network',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        plot_bgcolor='white'
    )
    
    return fig

def create_metrics_visualizations(report):
    """Create various metrics visualizations."""
    # Node type distribution
    if 'metrics' in report and 'node_frequencies' in report['metrics']:
        node_freq_df = pd.DataFrame([
            {'Node Type': k, 'Count': v} 
            for k, v in report['metrics']['node_frequencies'].items()
        ])
        node_freq_chart = px.bar(
            node_freq_df, 
            x='Node Type', 
            y='Count',
            title='Distribution of Code Elements'
        )
    else:
        node_freq_chart = None
    
    # Class hierarchy visualization
    class_hierarchy = report.get('class_hierarchy', {})
    if class_hierarchy:
        hierarchy_data = []
        for file, classes in class_hierarchy.items():
            for class_name, inheritance in classes.items():
                if inheritance:
                    for parent in inheritance:
                        hierarchy_data.append({
                            'Child': class_name,
                            'Parent': parent
                        })
        if hierarchy_data:
            hierarchy_df = pd.DataFrame(hierarchy_data)
            hierarchy_chart = px.treemap(
                hierarchy_df,
                path=['Parent', 'Child'],
                title='Class Hierarchy'
            )
        else:
            hierarchy_chart = None
    else:
        hierarchy_chart = None
        
    return node_freq_chart, hierarchy_chart

def main():
    st.set_page_config(page_title="GitHub Code Analyzer", layout="wide")
    
    st.title("🔍 GitHub Code Analyzer")
    st.markdown("""
    Analyze any Python GitHub repository to understand its structure, dependencies, and metrics.
    Just enter the repository URL below!
    """)
    
    # Input for GitHub repository URL
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/Spyrosigma/Code-Analyzer"
    )
    
    # Custom ignore directories
    st.sidebar.header("Analysis Settings")
    default_ignore = {'venv', 'env', '__pycache__', '.git', 'node_modules'}
    custom_ignore = st.sidebar.text_area(
        "Directories to ignore (one per line)",
        value='\n'.join(default_ignore)
    )
    ignore_dirs = set(custom_ignore.split('\n'))
    
    if st.button("Analyze Repository"):
        if not repo_url:
            st.warning("Please enter a GitHub repository URL")
            return
            
        with st.spinner("Cloning repository..."):
            repo_dir = clone_github_repo(repo_url)
            
        if repo_dir:
            try:
                with st.spinner("Analyzing code..."):
                    # Show progress bar
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    # Run analysis
                    analyze_codebase(repo_dir, ignore_dirs)
                    
                    # Load results
                    with open('analysis_report.json', 'r') as f:
                        report = json.load(f)
                    with open('code_chunks.json', 'r') as f:
                        chunks = json.load(f)
                    
                    # Update progress
                    progress_bar.progress(100)
                    progress_text.text("Analysis complete!")
                    time.sleep(1)
                    progress_bar.empty()
                    progress_text.empty()
                    
                    # Display results in tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Metrics", 
                        "🔗 Dependencies",
                        "📑 Code Chunks",
                        "📝 Raw Report"
                    ])
                    
                    with tab1:
                        st.header("Code Metrics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Functions", report['total_functions'])
                            st.metric("Total Classes", report['total_classes'])
                        
                        # Create visualizations
                        node_freq_chart, hierarchy_chart = create_metrics_visualizations(report)
                        
                        if node_freq_chart:
                            st.plotly_chart(node_freq_chart, use_container_width=True)
                        
                        if hierarchy_chart:
                            st.plotly_chart(hierarchy_chart, use_container_width=True)
                    
                    with tab2:
                        st.header("Dependency Network")
                        G = create_dependency_network(report)
                        fig = plot_dependency_network(G)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional network metrics
                        st.subheader("Network Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Dependencies", G.number_of_edges())
                        with col2:
                            st.metric("Connected Components", nx.number_connected_components(G.to_undirected()))
                        with col3:
                            st.metric("Average Connections", round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2))
                    
                    with tab3:
                        st.header("Code Chunks")
                        for i, chunk in enumerate(chunks):
                            with st.expander(f"Chunk {i+1}: {chunk.get('type', 'Unknown')}"):
                                st.json(chunk)
                    
                    with tab4:
                        st.header("Raw Analysis Report")
                        st.json(report)
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
            finally:
                # Cleanup
                shutil.rmtree(repo_dir)

if __name__ == "__main__":
    main()