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
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard - Análise de Estresse Estudantil",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
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
        st.error("❌ Arquivo 'StressLevelDataset.csv' não encontrado!")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Dashboard - Análise de Estresse Estudantil</h1>', unsafe_allow_html=True)
    
    # Carregar dados
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar com filtros
    st.sidebar.header("🎛️ Filtros Interativos")
    
    # Filtros
    stress_range = st.sidebar.slider(
        "📈 Nível de Estresse",
        min_value=int(df['stress_level'].min()),
        max_value=int(df['stress_level'].max()),
        value=(int(df['stress_level'].min()), int(df['stress_level'].max()))
    )
    
    bullying_filter = st.sidebar.multiselect(
        "🎭 Status de Bullying",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: "Sem Bullying" if x == 0 else "Com Bullying"
    )
    
    anxiety_range = st.sidebar.slider(
        "😰 Nível de Ansiedade",
        min_value=float(df['anxiety_level'].min()),
        max_value=float(df['anxiety_level'].max()),
        value=(float(df['anxiety_level'].min()), float(df['anxiety_level'].max()))
    )
    
    # Aplicar filtros
    filtered_df = df[
        (df['stress_level'] >= stress_range[0]) & 
        (df['stress_level'] <= stress_range[1]) &
        (df['bullying'].isin(bullying_filter)) &
        (df['anxiety_level'] >= anxiety_range[0]) &
        (df['anxiety_level'] <= anxiety_range[1])
    ]
    
    st.sidebar.markdown(f"📊 **Dados filtrados:** {len(filtered_df)} de {len(df)} estudantes")
    
    # === SEÇÃO 1: MÉTRICAS PRINCIPAIS ===
    st.header("📊 Visão Geral - Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_stress = filtered_df['stress_level'].mean()
        st.metric(
            label="📈 Estresse Médio",
            value=f"{avg_stress:.2f}",
            delta=f"{avg_stress - df['stress_level'].mean():.2f}" if len(filtered_df) < len(df) else None
        )
    
    with col2:
        bullying_pct = (filtered_df['bullying'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric(
            label="🎭 % com Bullying",
            value=f"{bullying_pct:.1f}%",
            delta=f"{bullying_pct - (df['bullying'].sum() / len(df)) * 100:.1f}%" if len(filtered_df) < len(df) else None
        )
    
    with col3:
        avg_selfesteem = filtered_df['self_esteem'].mean()
        st.metric(
            label="💪 Autoestima Média",
            value=f"{avg_selfesteem:.2f}",
            delta=f"{avg_selfesteem - df['self_esteem'].mean():.2f}" if len(filtered_df) < len(df) else None
        )
    
    with col4:
        high_stress_pct = (len(filtered_df[filtered_df['stress_level'] >= 3]) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.metric(
            label="🚨 % Alto Estresse",
            value=f"{high_stress_pct:.1f}%",
            delta=f"{high_stress_pct - (len(df[df['stress_level'] >= 3]) / len(df)) * 100:.1f}%" if len(filtered_df) < len(df) else None
        )
    
    # === SEÇÃO 2: DISTRIBUIÇÕES ===
    st.header("📈 Distribuição dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de Estresse
        fig_hist = px.histogram(
            filtered_df, 
            x='stress_level',
            nbins=20,
            title="📊 Distribuição dos Níveis de Estresse",
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(
            xaxis_title="Nível de Estresse",
            yaxis_title="Frequência",
            showlegend=False
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot por bullying
        fig_box = px.box(
            filtered_df,
            x='bullying',
            y='stress_level',
            title="📦 Estresse por Status de Bullying",
            color='bullying',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig_box.update_xaxes(
            tickvals=[0, 1],
            ticktext=['Sem Bullying', 'Com Bullying']
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # === SEÇÃO 3: ANÁLISE DE CORRELAÇÕES ===
    st.header("🔗 Análise de Correlações")
    
    # Calcular matriz de correlação
    correlation_vars = ['stress_level', 'anxiety_level', 'self_esteem', 'sleep_quality', 
                       'bullying', 'depression', 'peer_pressure', 'social_support']
    corr_matrix = filtered_df[correlation_vars].corr()
    
    # Heatmap de correlações
    fig_heatmap = px.imshow(
        corr_matrix,
        title="🌡️ Matriz de Correlações entre Variáveis",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        text_auto=True
    )
    fig_heatmap.update_layout(
        width=800,
        height=600
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Top correlações com estresse
    stress_corr = corr_matrix['stress_level'].abs().sort_values(ascending=False)
    stress_corr = stress_corr[stress_corr.index != 'stress_level']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Fatores Mais Correlacionados com Estresse")
        for var, corr in stress_corr.head(5).items():
            direction = "📈 Positiva" if corr_matrix.loc['stress_level', var] > 0 else "📉 Negativa"
            st.markdown(f"**{var.replace('_', ' ').title()}:** {corr:.3f} {direction}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Gráfico de barras das correlações
        fig_corr_bar = px.bar(
            x=stress_corr.head(5).values,
            y=stress_corr.head(5).index,
            orientation='h',
            title="📊 Top 5 Correlações com Estresse",
            color=stress_corr.head(5).values,
            color_continuous_scale="viridis"
        )
        fig_corr_bar.update_layout(
            yaxis_title="Variáveis",
            xaxis_title="Correlação (valor absoluto)",
            showlegend=False
        )
        st.plotly_chart(fig_corr_bar, use_container_width=True)
    
    # === SEÇÃO 4: IMPACTO DO BULLYING ===
    st.header("🎭 Análise Detalhada do Impacto do Bullying")
    
    # Comparação entre grupos
    no_bullying = filtered_df[filtered_df['bullying'] == 0]
    with_bullying = filtered_df[filtered_df['bullying'] == 1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 😊 Sem Bullying")
        if len(no_bullying) > 0:
            st.markdown(f"**👥 Estudantes:** {len(no_bullying)}")
            st.markdown(f"**📈 Estresse Médio:** {no_bullying['stress_level'].mean():.2f}")
            st.markdown(f"**😰 Ansiedade Média:** {no_bullying['anxiety_level'].mean():.1f}")
            st.markdown(f"**💪 Autoestima Média:** {no_bullying['self_esteem'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 😰 Com Bullying")
        if len(with_bullying) > 0:
            st.markdown(f"**👥 Estudantes:** {len(with_bullying)}")
            st.markdown(f"**📈 Estresse Médio:** {with_bullying['stress_level'].mean():.2f}")
            st.markdown(f"**😰 Ansiedade Média:** {with_bullying['anxiety_level'].mean():.1f}")
            st.markdown(f"**💪 Autoestima Média:** {with_bullying['self_esteem'].mean():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Diferença")
        if len(no_bullying) > 0 and len(with_bullying) > 0:
            stress_diff = with_bullying['stress_level'].mean() - no_bullying['stress_level'].mean()
            anxiety_diff = with_bullying['anxiety_level'].mean() - no_bullying['anxiety_level'].mean()
            esteem_diff = with_bullying['self_esteem'].mean() - no_bullying['self_esteem'].mean()
            
            st.markdown(f"**📈 Δ Estresse:** {stress_diff:+.2f}")
            st.markdown(f"**😰 Δ Ansiedade:** {anxiety_diff:+.1f}")
            st.markdown(f"**💪 Δ Autoestima:** {esteem_diff:+.2f}")
            
            # Teste estatístico
            if len(no_bullying) > 1 and len(with_bullying) > 1:
                t_stat, p_value = stats.ttest_ind(with_bullying['stress_level'], no_bullying['stress_level'])
                significance = "Significativa" if p_value < 0.05 else "Não significativa"
                st.markdown(f"**🔬 Diferença:** {significance}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos comparativos
    col1, col2 = st.columns(2)
    
    with col1:
        # Violin plot
        fig_violin = px.violin(
            filtered_df,
            x='bullying',
            y='stress_level',
            box=True,
            title="🎻 Distribuição de Estresse por Bullying",
            color='bullying',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig_violin.update_xaxes(
            tickvals=[0, 1],
            ticktext=['Sem Bullying', 'Com Bullying']
        )
        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with col2:
        # Múltiplas métricas
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
            title="📊 Comparação de Múltiplas Métricas",
            color_discrete_map={'Sem Bullying': '#2ecc71', 'Com Bullying': '#e74c3c'}
        )
        fig_comparison.update_xaxes(tickangle=45)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # === SEÇÃO 5: ANÁLISE MULTIVARIADA ===
    st.header("🔍 Análise Multivariada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot 3D
        fig_3d = px.scatter_3d(
            filtered_df,
            x='anxiety_level',
            y='self_esteem',
            z='stress_level',
            color='bullying',
            title="🌐 Análise Tridimensional",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            hover_data=['sleep_quality', 'depression']
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        # Scatter plot com tamanho
        fig_scatter = px.scatter(
            filtered_df,
            x='anxiety_level',
            y='stress_level',
            size='depression',
            color='bullying',
            title="💫 Ansiedade vs Estresse (tamanho = depressão)",
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            hover_data=['self_esteem', 'sleep_quality']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # === SEÇÃO 6: MODELAGEM PREDITIVA ===
    st.header("🤖 Modelo Preditivo de Estresse")
    
    # Preparar dados para o modelo
    feature_cols = ['anxiety_level', 'future_career_concerns', 'bullying', 
                   'depression', 'sleep_quality', 'peer_pressure']
    
    if all(col in filtered_df.columns for col in feature_cols):
        X = filtered_df[feature_cols]
        y = filtered_df['stress_level']
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar modelo
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Fazer predições
        y_pred = model.predict(X_test)
        
        # Métricas do modelo
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 R² Score", f"{r2:.3f}")
        
        with col2:
            st.metric("📏 RMSE", f"{rmse:.3f}")
        
        with col3:
            accuracy_category = "Excelente" if r2 > 0.8 else "Bom" if r2 > 0.6 else "Regular"
            st.metric("🎯 Performance", accuracy_category)
        
        # Importância das features
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
                title="📊 Importância das Variáveis",
                color='Coefficient',
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Predições vs Real
            fig_pred = px.scatter(
                x=y_test,
                y=y_pred,
                title="🎯 Predições vs Valores Reais",
                labels={'x': 'Valores Reais', 'y': 'Predições'}
            )
            fig_pred.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Linha Perfeita',
                line=dict(dash='dash', color='red')
            ))
            st.plotly_chart(fig_pred, use_container_width=True)
    
    # === SEÇÃO 7: INSIGHTS E RECOMENDAÇÕES ===
    st.header("💡 Insights e Recomendações")
    
    insights = []
    
    # Análise do bullying
    if len(filtered_df) > 0:
        bullying_rate = (filtered_df['bullying'].sum() / len(filtered_df)) * 100
        if bullying_rate > 30:
            insights.append("🚨 **Alta prevalência de bullying** - Implementar programa de prevenção urgente")
        elif bullying_rate > 15:
            insights.append("⚠️ **Bullying moderado** - Fortalecer políticas anti-bullying")
        else:
            insights.append("✅ **Baixa incidência de bullying** - Manter programas preventivos")
    
    # Análise do estresse
    if len(filtered_df) > 0:
        high_stress_rate = (len(filtered_df[filtered_df['stress_level'] >= 3]) / len(filtered_df)) * 100
        if high_stress_rate > 25:
            insights.append("😰 **Alto nível de estresse** - Implementar técnicas de gestão de estresse")
        elif high_stress_rate > 15:
            insights.append("🟡 **Estresse moderado** - Oferecer suporte psicológico")
        else:
            insights.append("😌 **Níveis de estresse controlados** - Continuar monitoramento")
    
    # Análise da autoestima
    if len(filtered_df) > 0:
        avg_self_esteem = filtered_df['self_esteem'].mean()
        if avg_self_esteem < 2.5:
            insights.append("💪 **Baixa autoestima** - Desenvolver programas de empoderamento")
        elif avg_self_esteem < 3.5:
            insights.append("🔄 **Autoestima moderada** - Incentivar atividades de desenvolvimento pessoal")
        else:
            insights.append("🌟 **Boa autoestima** - Manter ambiente positivo")
    
    # Mostrar insights
    for i, insight in enumerate(insights, 1):
        st.markdown(f'<div class="insight-box">{i}. {insight}</div>', unsafe_allow_html=True)
    
    # === SEÇÃO 8: DADOS FILTRADOS ===
    with st.expander("📋 Ver Dados Filtrados", expanded=False):
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download dos dados filtrados
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Download dados filtrados (CSV)",
            data=csv,
            file_name="dados_filtrados_estresse.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("### 📈 Dashboard de Análise de Estresse Estudantil")
    st.markdown("*Desenvolvido para apoiar a tomada de decisões baseada em dados na educação*")

if __name__ == "__main__":
    main()
