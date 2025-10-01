import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import f_oneway, pearsonr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard - Análise de Estresse Estudantil",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Estilo principal */
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        color: #1e3a8a;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e0f2fe 0%, #f8fafc 100%);
        border-radius: 15px;
        border-left: 6px solid #1e3a8a;
    }
    
    /* Cards de métricas melhorados */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
        color: #1e293b;
    }
    
    .metric-card h3 {
        color: #1e40af;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .metric-card strong {
        color: #0f172a;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Caixas de insights melhoradas */
    .insight-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
        color: #1e293b;
        font-weight: 500;
    }
    
    .insight-box h4 {
        color: #1e40af;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .insight-box p {
        color: #374151;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    /* Seções com melhor espaçamento */
    .section-divider {
        margin: 3rem 0 2rem 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #10b981, #3b82f6);
        border-radius: 2px;
    }
    
    /* Estilo para títulos de seção */
    .section-header {
        color: #1e40af;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Container principal com padding adequado */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar melhorada */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Métricas do Streamlit customizadas */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Remover espaçamento excessivo */
    .element-container {
        margin-bottom: 1rem !important;
    }
    
    /* Estilo para tabelas */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Estilo para insights individuais */
    .insight-item {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-left: 4px solid #059669;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #111827;
        font-weight: 500;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .insight-item:hover {
        background: #f9fafb;
        transform: translateX(2px);
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega e processa os dados"""
    try:
        df = pd.read_csv("StressLevelDataset.csv")
        return df
    except FileNotFoundError:
        st.error(" Arquivo 'StressLevelDataset.csv' não encontrado!")
        return None

def main():
    st.markdown('<h1 class="main-header"> Dashboard - Análise de Estresse Estudantil</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df is None:
        st.stop()
    
    st.sidebar.markdown("## Painel de Controle")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Navegação Rápida")
    nav_options = {
        "Métricas Principais": "#m-tricas-principais",
        "Distribuições": "#distribui-o-dos-dados", 
        "Correlações": "#an-lise-de-correla-es",
        "Impacto Bullying": "#an-lise-detalhada-do-impacto-do-bullying",
        "Análise Multivariada": "#an-lise-multivariada",
        "Testes Estatísticos": "#testes-de-hip-tese-estat-stica",
        "Modelo Preditivo": "#modelo-preditivo-de-estresse",
        "Insights": "#insights-e-recomenda-es"
    }
    
    selected_section = st.sidebar.selectbox(
        "Ir para seção:",
        options=list(nav_options.keys()),
        help="Navegue rapidamente para diferentes seções do dashboard"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Filtros de Dados")
    
    stress_range = st.sidebar.slider(
        "Nível de Estresse",
        min_value=int(df['stress_level'].min()),
        max_value=int(df['stress_level'].max()),
        value=(int(df['stress_level'].min()), int(df['stress_level'].max())),
        help="Filtre por faixa de nível de estresse"
    )
    
    bullying_filter = st.sidebar.multiselect(
        "Status de Bullying",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: "Sem Bullying" if x == 0 else "Com Bullying",
        help="Inclua ou exclua estudantes com histórico de bullying"
    )
    
    anxiety_range = st.sidebar.slider(
        "Nível de Ansiedade",
        min_value=float(df['anxiety_level'].min()),
        max_value=float(df['anxiety_level'].max()),
        value=(float(df['anxiety_level'].min()), float(df['anxiety_level'].max())),
        help="Filtre por faixa de nível de ansiedade"
    )
    
    st.sidebar.markdown("### Filtros Adicionais")
    
    sleep_quality_filter = st.sidebar.multiselect(
        "Qualidade do Sono",
        options=sorted(df['sleep_quality'].unique()),
        default=sorted(df['sleep_quality'].unique()),
        help="Selecione níveis de qualidade do sono"
    )
    
    depression_range = st.sidebar.slider(
        "Nível de Depressão",
        min_value=int(df['depression'].min()),
        max_value=int(df['depression'].max()),
        value=(int(df['depression'].min()), int(df['depression'].max())),
        help="Filtre por faixa de nível de depressão"
    )
    
    filtered_df = df[
        (df['stress_level'] >= stress_range[0]) & 
        (df['stress_level'] <= stress_range[1]) &
        (df['bullying'].isin(bullying_filter)) &
        (df['anxiety_level'] >= anxiety_range[0]) &
        (df['anxiety_level'] <= anxiety_range[1]) &
        (df['sleep_quality'].isin(sleep_quality_filter)) &
        (df['depression'] >= depression_range[0]) &
        (df['depression'] <= depression_range[1])
    ]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Resumo dos Dados")
    
    total_students = len(df)
    filtered_students = len(filtered_df)
    percentage = (filtered_students / total_students * 100) if total_students > 0 else 0
    
    st.sidebar.metric("Total de Estudantes", f"{total_students:,}")
    st.sidebar.metric("Dados Filtrados", f"{filtered_students:,}")
    st.sidebar.metric("Percentual", f"{percentage:.1f}%")
    
    if filtered_students > 0:
        avg_stress_filtered = filtered_df['stress_level'].mean()
        avg_stress_total = df['stress_level'].mean()
        stress_diff = avg_stress_filtered - avg_stress_total
        
        st.sidebar.metric(
            "Estresse Médio (Filtrado)", 
            f"{avg_stress_filtered:.2f}",
            delta=f"{stress_diff:+.2f}"
        )
    
    st.header("Visão Geral - Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_stress = filtered_df['stress_level'].mean()
        st.metric(
            label="Estresse Médio",
            value=f"{avg_stress:.2f}",
            delta=f"{avg_stress - df['stress_level'].mean():.2f}" if len(filtered_df) < len(df) else None
        )
    
    with col2:
        bullying_pct = (filtered_df['bullying'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric(
            label="% com Bullying",
            value=f"{bullying_pct:.1f}%",
            delta=f"{bullying_pct - (df['bullying'].sum() / len(df)) * 100:.1f}%" if len(filtered_df) < len(df) else None
        )
    
    with col3:
        avg_selfesteem = filtered_df['self_esteem'].mean()
        st.metric(
            label="Autoestima Média",
            value=f"{avg_selfesteem:.2f}",
            delta=f"{avg_selfesteem - df['self_esteem'].mean():.2f}" if len(filtered_df) < len(df) else None
        )
    
    with col4:
        high_stress_pct = (len(filtered_df[filtered_df['stress_level'] >= 3]) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric(
            label="% Alto Estresse",
            value=f"{high_stress_pct:.1f}%",
            delta=f"{high_stress_pct - (len(df[df['stress_level'] >= 3]) / len(df)) * 100:.1f}%" if len(filtered_df) < len(df) else None
        )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Distribuição dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            filtered_df, 
            x='stress_level',
            nbins=20,
            title="Distribuição dos Níveis de Estresse",
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(
            xaxis_title="Nível de Estresse",
            yaxis_title="Frequência",
            showlegend=False
        )
        st.plotly_chart(
            fig_hist, 
            use_container_width=True,
            config={"responsive": True}
        )
    
    with col2:
        fig_box = px.box(
            filtered_df,
            x='bullying',
            y='stress_level',
            title="Estresse por Status de Bullying",
            color='bullying',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig_box.update_xaxes(
            tickvals=[0, 1],
            ticktext=['Sem Bullying', 'Com Bullying']
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(
            fig_box, 
            use_container_width=True,
            config={"responsive": True}
        )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Análise de Correlações")
    
    st.info("**Correlações calculadas com dataset completo** para análise estatística precisa. Filtros aplicam-se apenas às outras visualizações.")
    
    correlation_vars = ['stress_level', 'anxiety_level', 'self_esteem', 'sleep_quality', 
                    'bullying', 'depression', 'peer_pressure', 'social_support']
    corr_matrix = df[correlation_vars].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        title="Matriz de Correlações entre Variáveis",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=True
    )
    fig_heatmap.update_layout(
        width=800,
        height=600
    )
    st.plotly_chart(
        fig_heatmap, 
        use_container_width=True,
        config={"responsive": True}
    )
    
    stress_corr = corr_matrix['stress_level'].abs().sort_values(ascending=False)
    stress_corr = stress_corr[stress_corr.index != 'stress_level']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### Fatores Mais Correlacionados com Estresse")
        for var, corr in stress_corr.head(5).items():
            direction = "Positiva" if corr_matrix.loc['stress_level', var] > 0 else "Negativa"
            st.markdown(f"**{var.replace('_', ' ').title()}:** {corr:.3f} {direction}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        fig_corr_bar = px.bar(
            x=stress_corr.head(5).values,
            y=stress_corr.head(5).index,
            orientation='h',
            title="Top 5 Correlações com Estresse",
            color=stress_corr.head(5).values,
            color_continuous_scale="viridis"
        )
        fig_corr_bar.update_layout(
            yaxis_title="Variáveis",
            xaxis_title="Correlação (valor absoluto)",
            showlegend=False
        )
        st.plotly_chart(
            fig_corr_bar, 
            use_container_width=True,
            config={"responsive": True}
        )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Análise Detalhada do Impacto do Bullying")
    
    no_bullying = filtered_df[filtered_df['bullying'] == 0]
    with_bullying = filtered_df[filtered_df['bullying'] == 1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Sem Bullying")
        if len(no_bullying) > 0:
            st.markdown(f"**Estudantes:** {len(no_bullying)}")
            st.markdown(f"**Estresse Médio:** {no_bullying['stress_level'].mean():.2f}")
            st.markdown(f"**Ansiedade Média:** {no_bullying['anxiety_level'].mean():.1f}")
            st.markdown(f"**Autoestima Média:** {no_bullying['self_esteem'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Com Bullying")
        if len(with_bullying) > 0:
            st.markdown(f"**Estudantes:** {len(with_bullying)}")
            st.markdown(f"**Estresse Médio:** {with_bullying['stress_level'].mean():.2f}")
            st.markdown(f"**Ansiedade Média:** {with_bullying['anxiety_level'].mean():.1f}")
            st.markdown(f"**Autoestima Média:** {with_bullying['self_esteem'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Diferença")
        if len(no_bullying) > 0 and len(with_bullying) > 0:
            stress_diff = with_bullying['stress_level'].mean() - no_bullying['stress_level'].mean()
            anxiety_diff = with_bullying['anxiety_level'].mean() - no_bullying['anxiety_level'].mean()
            esteem_diff = with_bullying['self_esteem'].mean() - no_bullying['self_esteem'].mean()
            
            st.markdown(f"**Estresse:** {stress_diff:+.2f}")
            st.markdown(f"**Ansiedade:** {anxiety_diff:+.1f}")
            st.markdown(f"**Autoestima:** {esteem_diff:+.2f}")
            
            if len(no_bullying) > 1 and len(with_bullying) > 1:
                t_stat, p_value = stats.ttest_ind(with_bullying['stress_level'], no_bullying['stress_level'])
                significance = "Significativa" if p_value < 0.05 else "Não significativa"
                st.markdown(f"**Diferença:** {significance}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_violin = px.violin(
            filtered_df,
            x='bullying',
            y='stress_level',
            box=True,
            title="Distribuição de Estresse por Bullying",
            color='bullying',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig_violin.update_xaxes(
            tickvals=[0, 1],
            ticktext=['Sem Bullying', 'Com Bullying']
        )
        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(
            fig_violin,
            use_container_width=True,   
            config={"responsive": True}
        )

    
    with col2:
        metrics = ['stress_level', 'anxiety_level', 'self_esteem', 'sleep_quality']
        bullying_comparison = []
        
        for metric in metrics:
            bullying_comparison.extend([
                {
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': no_bullying[metric].mean() if len(no_bullying) > 0 else 0,
                    'Group': 'Sem Bullying'
                },
                {
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': with_bullying[metric].mean() if len(with_bullying) > 0 else 0,
                    'Group': 'Com Bullying'
                }
            ])
        
        comparison_df = pd.DataFrame(bullying_comparison)
        
        fig_comparison = px.bar(
            comparison_df,
            x='Metric',
            y='Value',
            color='Group',
            barmode='group',
            title="Comparação de Múltiplas Métricas",
            color_discrete_map={'Sem Bullying': '#2ecc71', 'Com Bullying': '#e74c3c'}
        )
        fig_comparison.update_xaxes(tickangle=45)
        st.plotly_chart(
            fig_comparison,
            use_container_width=True,
            config={"responsive": True}
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Análise Multivariada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_3d = px.scatter_3d(
            filtered_df,
            x='anxiety_level',
            y='self_esteem',
            z='stress_level',
            color='bullying',
            title="Análise Tridimensional",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            hover_data=['sleep_quality', 'depression']
        )
        st.plotly_chart(
            fig_3d, 
            use_container_width=True,
            config={"responsive": True}
        )
    
    with col2:
        fig_scatter = px.scatter(
            filtered_df,
            x='anxiety_level',
            y='stress_level',
            size='depression',
            color='bullying',
            title="Ansiedade vs Estresse (tamanho = depressão)",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            hover_data=['self_esteem', 'sleep_quality']
        )
        st.plotly_chart(
            fig_scatter, 
            use_container_width=True,
            config={"responsive": True}
        )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Testes de Hipótese Estatística")
    
    st.markdown("""
    <div class="insight-box">
    <h4>Análises Estatísticas Realizadas</h4>
    <p><strong>Esta seção apresenta testes estatísticos rigorosos para validar hipóteses sobre fatores que influenciam o estresse estudantil.</strong></p>
    <p>• <strong>ANOVA:</strong> Compara médias de estresse entre diferentes níveis de qualidade do sono</p>
    <p>• <strong>Correlação de Pearson:</strong> Analisa a relação linear entre carga de estudos e nível de estresse</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Hipótese 1: ANOVA - Qualidade do Sono vs Nível de Estresse")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        grupos_sleep = []
        sleep_quality_stats = []
        
        for i in sorted(filtered_df['sleep_quality'].unique()):
            grupo = filtered_df[filtered_df['sleep_quality'] == i]['stress_level']
            grupos_sleep.append(grupo)
            sleep_quality_stats.append({
                'Sleep Quality': i,
                'Count': len(grupo),
                'Mean': grupo.mean(),
                'Std': grupo.std()
            })
        
        fig_anova = px.box(
            filtered_df,
            x='sleep_quality',
            y='stress_level',
            title="Distribuição de Stress Level por Sleep Quality",
            color='sleep_quality',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        for i, stat in enumerate(sleep_quality_stats):
            fig_anova.add_annotation(
                x=stat['Sleep Quality'],
                y=stat['Mean'],
                text=f"μ={stat['Mean']:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                bgcolor="yellow",
                bordercolor="red",
                borderwidth=1
            )
        
        fig_anova.update_layout(
            xaxis_title="Sleep Quality",
            yaxis_title="Stress Level",
            showlegend=False
        )
        st.plotly_chart(
            fig_anova, 
            use_container_width=True,
            config={"responsive": True}
        )
    
    with col2:
        if len(grupos_sleep) > 1 and all(len(grupo) > 1 for grupo in grupos_sleep):
            f_stat, p_value_anova = f_oneway(*grupos_sleep)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Resultados ANOVA")
            st.markdown(f"**F-statistic:** {f_stat:.4f}")
            st.markdown(f"**p-valor:** {p_value_anova:.2e}")
            
            if p_value_anova < 0.05:
                st.markdown("**SIGNIFICATIVO**")
                st.markdown("Rejeitamos H₀: Há diferença significativa entre as médias")
            else:
                st.markdown("**NÃO SIGNIFICATIVO**")
                st.markdown("Não rejeitamos H₀: Não há evidência de diferença")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("#### Estatísticas por Grupo")
            stats_df = pd.DataFrame(sleep_quality_stats)
            st.dataframe(stats_df, width="stretch")
    
    st.markdown("---")
    
    st.subheader("Hipótese 2: Correlação - Carga de Estudos vs Nível de Estresse")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_corr = px.scatter(
            filtered_df,
            x='study_load',
            y='stress_level',
            title="Correlação entre Study Load e Stress Level",
            opacity=0.6,
            color='bullying',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            hover_data=['anxiety_level', 'sleep_quality']
        )
        
        if len(filtered_df) > 1:
            z = np.polyfit(filtered_df['study_load'], filtered_df['stress_level'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(filtered_df['study_load'].min(), filtered_df['study_load'].max(), 100)
            
            fig_corr.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                name='Linha de Regressão',
                line=dict(color='red', width=3, dash='dash')
            ))
        
        fig_corr.update_layout(
            xaxis_title="Study Load",
            yaxis_title="Stress Level"
        )
        st.plotly_chart(
            fig_corr,
            use_container_width=True,
            config={"responsive": True}
        )
    
    with col2:
        if len(filtered_df) > 2:
            correlation_coef, p_value_corr = pearsonr(filtered_df['stress_level'], filtered_df['study_load'])
            r_squared = correlation_coef ** 2
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### Resultados Correlação")
            st.markdown(f"**Coeficiente (r):** {correlation_coef:.4f}")
            st.markdown(f"**p-valor:** {p_value_corr:.2e}")
            st.markdown(f"**R² (variância explicada):** {r_squared:.1%}")
            
            if abs(correlation_coef) < 0.3:
                forca = "Fraca"
                emoji = "🟡"
            elif abs(correlation_coef) < 0.7:
                forca = "Moderada"
                emoji = "🟠"
            else:
                forca = "Forte"
                emoji = "🔴"
            
            st.markdown(f"**Força:** {emoji} {forca}")
            
            if p_value_corr < 0.05:
                st.markdown("**SIGNIFICATIVO**")
                st.markdown("Correlação positiva confirmada")
            else:
                st.markdown("**NÃO SIGNIFICATIVO**")
                st.markdown("Correlação não confirmada")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Resumo dos Testes de Hipótese")
    
    summary_data = []
    
    if len(grupos_sleep) > 1 and all(len(grupo) > 1 for grupo in grupos_sleep):
        summary_data.append({
            'Teste': 'ANOVA (Sleep Quality → Stress)',
            'Estatística': f'F = {f_stat:.3f}',
            'p-valor': f'{p_value_anova:.2e}',
            'Resultado': 'Significativo' if p_value_anova < 0.05 else '❌ Não significativo',
            'Interpretação': 'Qualidade do sono afeta o estresse' if p_value_anova < 0.05 else 'Sem evidência de efeito'
        })
    
    if len(filtered_df) > 2:
        summary_data.append({
            'Teste': 'Correlação (Study Load ↔ Stress)',
            'Estatística': f'r = {correlation_coef:.3f}',
            'p-valor': f'{p_value_corr:.2e}',
            'Resultado': 'Significativo' if p_value_corr < 0.05 else '❌ Não significativo',
            'Interpretação': f'Correlação {forca.lower()} positiva' if p_value_corr < 0.05 else 'Sem correlação significativa'
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, width="stretch")
    
    st.markdown("---")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Modelo Preditivo de Estresse")
    
    st.info("**Modelo treinado com dataset completo** (1000 amostras) para máxima precisão. Filtros aplicam-se apenas às visualizações.")
    
    feature_cols = ['anxiety_level', 'future_career_concerns', 'bullying', 
                'depression', 'sleep_quality', 'peer_pressure']
    
    if all(col in df.columns for col in feature_cols):
        X = df[feature_cols]
        y = df['stress_level']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score", f"{r2:.3f}")
        
        with col2:
            st.metric("RMSE", f"{rmse:.3f}")
        
        with col3:
            accuracy_category = "Excelente" if r2 > 0.8 else "Bom" if r2 > 0.6 else "Regular"
            st.metric("Performance", accuracy_category)
        
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_
        })
        feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_importance = px.bar(
                feature_importance,
                x='Abs_Coefficient',
                y='Feature',
                orientation='h',
                title="Importância das Variáveis",
                color='Coefficient',
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(
                fig_importance, 
                use_container_width=True,
                config={"responsive": True}
            )
        
        with col2:
            fig_pred = px.scatter(
                x=y_test,
                y=y_pred,
                title="Predições vs Valores Reais",
                labels={'x': 'Valores Reais', 'y': 'Predições'}
            )
            fig_pred.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Linha Perfeita',
                line=dict(dash='dash', color='red')
            ))
            st.plotly_chart(
                fig_pred,
                use_container_width=True,
                config={"responsive": True}
            )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.header("Insights e Recomendações")
    
    insights = []
    
    if len(filtered_df) > 0:
        bullying_rate = (filtered_df['bullying'].sum() / len(filtered_df)) * 100
        if bullying_rate > 30:
            insights.append("**Alta prevalência de bullying** - Implementar programa de prevenção urgente")
        elif bullying_rate > 15:
            insights.append("**Bullying moderado** - Fortalecer políticas anti-bullying")
        else:
            insights.append("**Baixa incidência de bullying** - Manter programas preventivos")
    
    if len(filtered_df) > 0:
        high_stress_rate = (len(filtered_df[filtered_df['stress_level'] >= 3]) / len(filtered_df)) * 100
        if high_stress_rate > 25:
            insights.append("**Alto nível de estresse** - Implementar técnicas de gestão de estresse")
        elif high_stress_rate > 15:
            insights.append("**Estresse moderado** - Oferecer suporte psicológico")
        else:
            insights.append("**Níveis de estresse controlados** - Continuar monitoramento")
    
    if len(filtered_df) > 0:
        avg_self_esteem = filtered_df['self_esteem'].mean()
        if avg_self_esteem < 2.5:
            insights.append("**Baixa autoestima** - Desenvolver programas de empoderamento")
        elif avg_self_esteem < 3.5:
            insights.append("**Autoestima moderada** - Incentivar atividades de desenvolvimento pessoal")
        else:
            insights.append("**Boa autoestima** - Manter ambiente positivo")
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f'<div class="insight-item">{i}. {insight}</div>', unsafe_allow_html=True)
    
    with st.expander("Ver Dados Filtrados", expanded=False):
        st.dataframe(filtered_df, width="stretch")
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download dados filtrados (CSV)",
            data=csv,
            file_name="dados_filtrados_estresse.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("### Dashboard de Análise de Estresse Estudantil")
    st.markdown("*Desenvolvido para apoiar a tomada de decisões baseada em dados na educação*")

if __name__ == "__main__":
    main()
