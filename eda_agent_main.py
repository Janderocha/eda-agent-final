"""
Agente Autônomo de EDA (Exploratory Data Analysis)
Framework: LangChain + Gemini (Google) + Streamlit
"""
import csv
import re
import io
from typing import Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importações do LangChain/Gemini
# CORREÇÃO: AgentExecutor foi movido para langchain_core.agents
from langchain.agents import AgentExecutor # CORREÇÃO: Importação corrigida
from langchain.agents import create_tool_calling_agent # Mantida em langchain.agents
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory # Importação da memória
from langchain_core.messages import HumanMessage, AIMessage # Importação correta para tipagem

# Configuração da página
st.set_page_config(page_title="Agente EDA Autônomo (Gemini)", layout="wide")

# Classe para gerenciar dados e análises
class DataAnalyzer:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.analysis_history = []
        self.conclusions = []

    def load_csv(self, file) -> str:
        """Carrega arquivo CSV com detecção automática de delimitador e robustez"""
        try:
            file.seek(0)
            sample = file.read(1024).decode('utf-8')
            file.seek(0)

            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            if delimiter not in [',', ';', '\t']:
                delimiter = ','

            self.df = pd.read_csv(
                file,
                delimiter=delimiter,
                on_bad_lines='skip',
                quoting=csv.QUOTE_MINIMAL,
                encoding='utf-8',
                engine='python'
            )

            info = f"""
    Arquivo carregado com sucesso!
    - Linhas (após limpeza): {len(self.df)}
    - Colunas: {len(self.df.columns)}
    - Colunas: {', '.join(self.df.columns.tolist())}
    - Delimitador detectado: '{delimiter}'
    """
            self.analysis_history.append({"action": "load", "result": info})
            return info

        except Exception as e:
            return f"Erro ao carregar arquivo: {str(e)}"

    def describe_data(self, columns: Optional[list] = None) -> str:
        """Descreve estatísticas dos dados"""
        if self.df is None:
            return "Nenhum arquivo carregado."

        try:
            if columns:
                desc = self.df[columns].describe(include='all').to_string()
            else:
                desc = self.df.describe(include='all').to_string()

            types_info = "\n\nTipos de dados:\n" + self.df.dtypes.to_string()
            result = desc + types_info
            self.analysis_history.append({"action": "describe", "result": result})
            return result
        except Exception as e:
            return f"Erro na análise descritiva: {str(e)}"

    def check_missing_values(self) -> str:
        """Verifica valores faltantes"""
        if self.df is None:
            return "Nenhum arquivo carregado."

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        result = pd.DataFrame({
            'Missing': missing,
            'Percentage': missing_pct
        }).to_string()

        self.analysis_history.append({"action": "missing", "result": result})
        return result

    def calculate_correlation(self, method: str = 'pearson') -> str:
        """Calcula matriz de correlação"""
        if self.df is None:
            return "Nenhum arquivo carregado."

        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return "Menos de duas colunas numéricas para calcular correlação."
            corr = self.df[numeric_cols].corr(method=method)
            result = corr.to_string()
            self.analysis_history.append({"action": "correlation", "result": result})
            return result
        except Exception as e:
            return f"Erro no cálculo de correlação: {str(e)}"

    def detect_outliers(self, column: str, method: str = 'iqr') -> str:
        """Detecta outliers em uma coluna"""
        if self.df is None:
            return "Nenhum arquivo carregado."

        try:
            if column not in self.df.columns:
                return f"Coluna '{column}' não encontrada."

            data = self.df[column].dropna()
            if not pd.api.types.is_numeric_dtype(data):
                return f"A coluna '{column}' não é numérica."

            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]

                result = f"""
Análise de Outliers (IQR) para '{column}':
- Limite inferior: {lower_bound:.2f}
- Limite superior: {upper_bound:.2f}
- Outliers detectados: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)
- Valores mín/máx dos outliers: {outliers.min():.2f} / {outliers.max():.2f}
"""
            else:
                z_scores = np.abs((data - data.mean()) / data.std())
                outliers = data[z_scores > 3]
                result = f"""
Análise de Outliers (Z-score) para '{column}':
- Outliers detectados (|z| > 3): {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)
"""

            self.analysis_history.append({"action": "outliers", "column": column, "result": result})
            return result
        except Exception as e:
            return f"Erro na detecção de outliers: {str(e)}"

    def generate_plot(self, plot_type: str, columns: list, **kwargs) -> str:
        """Gera gráficos"""
        if self.df is None:
            return "Nenhum arquivo carregado."

        try:
            if not columns:
                return "Nenhuma coluna especificada para o gráfico."

            fig, ax = plt.subplots(figsize=(10, 6))

            if plot_type == 'histogram':
                if columns[0] not in self.df.columns:
                    return f"Coluna '{columns[0]}' não encontrada."
                self.df[columns[0]].hist(bins=kwargs.get('bins', 30), ax=ax)
                ax.set_title(f'Histograma - {columns[0]}')
                ax.set_xlabel(columns[0])
                ax.set_ylabel('Frequência')

            elif plot_type == 'scatter':
                if len(columns) < 2:
                    return "Gráfico de dispersão requer duas colunas."
                if columns[0] not in self.df.columns or columns[1] not in self.df.columns:
                    return "Uma ou mais colunas não encontradas."
                ax.scatter(self.df[columns[0]], self.df[columns[1]], alpha=0.5)
                ax.set_xlabel(columns[0])
                ax.set_ylabel(columns[1])
                ax.set_title(f'Dispersão: {columns[0]} vs {columns[1]}')

            elif plot_type == 'boxplot':
                valid_cols = [c for c in columns if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]
                if not valid_cols:
                    return "Nenhuma coluna numérica válida para boxplot."
                self.df[valid_cols].boxplot(ax=ax)
                ax.set_title('Boxplot')
                ax.set_ylabel('Valores')

            elif plot_type == 'correlation':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    return "Menos de duas colunas numéricas para correlação."
                corr = self.df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title('Matriz de Correlação')

            elif plot_type == 'bar':
                if columns[0] not in self.df.columns:
                    return f"Coluna '{columns[0]}' não encontrada."
                value_counts = self.df[columns[0]].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f'Top 10 Valores - {columns[0]}')
                ax.set_ylabel('Contagem')

            else:
                return f"Tipo de gráfico '{plot_type}' não suportado."

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            st.image(buf, use_column_width=True)

            result = f"Gráfico '{plot_type}' gerado com sucesso para: {', '.join(columns)}"
            self.analysis_history.append({"action": "plot", "type": plot_type, "columns": columns})
            return result

        except Exception as e:
            return f"Erro ao gerar gráfico: {str(e)}"

    def add_conclusion(self, conclusion: str):
        """Adiciona conclusão à memória"""
        self.conclusions.append(conclusion)

    def get_conclusions_summary(self) -> str:
        """Retorna resumo das conclusões"""
        if not self.conclusions:
            return "Nenhuma conclusão registrada ainda."
        return "CONCLUSÕES ACUMULADAS:\n\n" + "\n\n".join([
            f"{i+1}. {c}" for i, c in enumerate(self.conclusions)
        ])

    def get_columns_list(self) -> List[str]:
        if self.df is None:
            return []
        return self.df.columns.tolist()


# Inicializa o analisador
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = DataAnalyzer()

if 'messages' not in st.session_state:
    st.session_state.messages = []

analyzer = st.session_state.analyzer

# Definição das Ferramentas (Tools)
# (As funções wrappers permanecem as mesmas, chamando os métodos do DataAnalyzer)

def tool_describe_data(query: str) -> str:
    """Obtém estatísticas descritivas completas do dataset"""
    return analyzer.describe_data()

def tool_check_missing(query: str) -> str:
    """Verifica valores faltantes em todas as colunas"""
    return analyzer.check_missing_values()

def tool_calculate_correlation(query: str) -> str:
    """Calcula matriz de correlação entre variáveis numéricas"""
    return analyzer.calculate_correlation()

def tool_detect_outliers(query: str) -> str:
    """Detecta outliers em uma coluna específica. Input: 'nome_coluna,metodo' (metodo: iqr ou zscore)"""
    parts = [p.strip() for p in re.split(r',', query)]
    column = parts[0]
    method = parts[1] if len(parts) > 1 else 'iqr'
    return analyzer.detect_outliers(column, method)

def tool_generate_plot(query: str) -> str:
    """Gera gráficos. Input: 'tipo,coluna1,coluna2'. Tipos: histogram, scatter, boxplot, correlation, bar"""
    parts = [p.strip() for p in re.split(r',', query)]
    plot_type = parts[0]
    columns = parts[1:] if len(parts) > 1 else []
    return analyzer.generate_plot(plot_type, columns)

def tool_get_columns(query: str) -> str:
    """Lista todas as colunas disponíveis no dataset"""
    if analyzer.df is None:
        return "Nenhum arquivo carregado."
    return f"Colunas disponíveis: {', '.join(analyzer.get_columns_list())}"

def tool_save_conclusion(conclusion: str) -> str:
    """Salva conclusões importantes da análise para referência futura"""
    analyzer.add_conclusion(conclusion)
    return "Conclusão salva com sucesso."

def tool_get_conclusions(query: str) -> str:
    """Recupera todas as conclusões salvas durante a análise"""
    return analyzer.get_conclusions_summary()


tools = [
    Tool(name="describe_data", func=tool_describe_data,
         description="Obtém estatísticas descritivas completas do dataset"),
    Tool(name="check_missing", func=tool_check_missing,
         description="Verifica valores faltantes em todas as colunas"),
    Tool(name="calculate_correlation", func=tool_calculate_correlation,
         description="Calcula matriz de correlação entre variáveis numéricas"),
    Tool(name="detect_outliers", func=tool_detect_outliers,
         description="Detecta outliers em uma coluna específica. Input: 'nome_coluna,metodo'"),
    Tool(name="generate_plot", func=tool_generate_plot,
         description="Gera gráficos. Input: 'tipo,coluna1,coluna2'. Tipos: histogram, scatter, boxplot, correlation, bar"),
    Tool(name="get_columns", func=tool_get_columns,
         description="Lista todas as colunas disponíveis no dataset"),
    Tool(name="save_conclusion", func=tool_save_conclusion,
         description="Salva conclusões importantes da análise para referência futura"),
    Tool(name="get_conclusions", func=tool_get_conclusions,
         description="Recupera todas as conclusões salvas durante a análise")
]

# Interface Streamlit
st.title("🤖 Agente Autônomo de EDA (Gemini)")
st.markdown("### Análise Exploratória de Dados com IA Google")

# Sidebar
with st.sidebar:
    st.header("📁 Carregar Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=['csv'])

    if uploaded_file:
        result = analyzer.load_csv(uploaded_file)
        st.success(result)

    st.markdown("---")
    st.header("⚙️ Configurações")
    gemini_api_key = st.text_input("Chave da API do Google (Gemini)", type="password",
                                   help="Obtenha em https://ai.google.dev/")

    model_choice = st.selectbox("Modelo Gemini", ["gemini-2.5-flash"], index=0)

    if st.button("🗑️ Limpar Histórico"):
        # Limpa o histórico de mensagens do Streamlit
        st.session_state.messages = []
        # A memória do LangChain será recriada na próxima execução, limpando-a implicitamente
        st.rerun()

# Área principal
if not uploaded_file:
    st.info("👈 Por favor, carregue um arquivo CSV na barra lateral para começar.")
elif not gemini_api_key:
    st.warning("⚠️ Insira sua chave da API do Gemini na barra lateral.")
else:
    # 1. Inicializa o LLM do Gemini
    llm = ChatGoogleGenerativeAI(
        model=model_choice,
        temperature=0,
        google_api_key=gemini_api_key,
    )
    # Prompt do sistema
    system_prompt = """Você é um especialista em análise exploratória de dados (EDA).
Seu objetivo é ajudar o usuário a entender seus dados através de análises estatísticas e visualizações.
INSTRUÇÕES IMPORTANTES:
1. Use as ferramentas disponíveis para analisar os dados.
2. Sempre que fizer uma análise importante, use a ferramenta 'save_conclusion' para salvar insights.
3. Para gráficos, use 'generate_plot' com os parâmetros corretos.
4. Seja objetivo e apresente números e estatísticas.
5. Quando perguntado sobre conclusões, use 'get_conclusions' para recuperar insights salvos.
6. Sempre comece verificando as colunas disponíveis com 'get_columns'.
Tipos de gráficos disponíveis:
- histogram: histograma de uma variável
- scatter: dispersão entre duas variáveis
- boxplot: boxplot de uma ou mais variáveis
- correlation: mapa de calor de correlação
- bar: gráfico de barras dos valores mais frequentes
Responda em **português do Brasil** de forma clara e didática."""

    # Cria o prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"), # Chave usada pela memória
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])

    # 2. Define a Memória
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5 # Mantém o contexto recente
    )

    # Cria o agente
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 3. Cria o AgentExecutor (COM MEMÓRIA)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
        memory=memory # INCLUSÃO DA MEMÓRIA
    )

    # Exibe histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do usuário
    if user_input := st.chat_input("Faça uma pergunta sobre os dados..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Invoca o agente
        with st.chat_message("assistant"):
            with st.spinner("Analisando com Gemini..."):
                try:
                    # O LangChain 1.0+ espera o 'input' para o executor e usa a memória
                    response = agent_executor.invoke({"input": user_input})
                    answer = response["output"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Erro ao processar com Gemini: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

                # Rodapé
st.markdown("---")
st.markdown("*Agente EDA com Gemini — IA para análise de dados*")