# 🤖 Agente Autônomo de EDA (Exploratory Data Analysis)

Este projeto implementa um Agente de Análise Exploratória de Dados (EDA) utilizando o framework **LangChain** (versão 0.2.x) e o modelo de linguagem **Gemini (Google)**. O agente permite que o usuário carregue um arquivo CSV e, através de comandos em linguagem natural, execute análises estatísticas, verificação de dados faltantes, detecção de outliers e gere visualizações de dados.

A interface gráfica é construída com **Streamlit**.

## ⚙️ Tecnologias Principais

* **Framework de Agente:** LangChain (Versão 0.2.x)
* **Modelo de Linguagem:** Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai`
* **Interface:** Streamlit
* **Análise de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn

---

## ✨ Funcionalidades do Agente (Tools)

O agente utiliza um conjunto de ferramentas Python robustas para interagir com o `DataFrame` carregado. O LLM decide qual ferramenta chamar, e com quais parâmetros, com base na solicitação do usuário.

| Ferramenta | Descrição | Parâmetros de Uso Típico |
| :--- | :--- | :--- |
| `get_columns` | Lista todas as colunas disponíveis no dataset. | Nenhum |
| `describe_data` | Retorna estatísticas descritivas completas do dataset. | Nenhum |
| `check_missing` | Verifica a contagem e porcentagem de valores faltantes por coluna. | Nenhum |
| `calculate_correlation`| Calcula a matriz de correlação entre colunas numéricas. | Nenhum (usa padrão 'pearson') |
| `detect_outliers` | Identifica outliers em uma coluna. | `nome_coluna,metodo` (métodos: iqr ou zscore) |
| `generate_plot` | **Gera e exibe gráficos** (histogram, scatter, boxplot, correlation, bar). | `tipo,coluna1,coluna2` (ex: `scatter,idade,salario`) |
| `save_conclusion` | Salva um insight importante gerado pela análise na memória. | `conclusion` (string) |
| `get_conclusions` | Recupera todas as conclusões salvas durante a sessão. | Nenhum |

---

## 🚀 Instalação e Execução

### 1. Pré-requisitos

Você precisará de uma chave de API válida do Google Gemini (`GEMINI_API_KEY`).

### 2. Criação do Ambiente

Crie e ative seu ambiente virtual (recomendado):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
```

### 3. Instalação de Dependências
Crie um arquivo requirements.txt com as dependências a seguir e instale-as:
```bash
# Dependências do Framework
streamlit>=1.36.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.9.0
seaborn>=0.13.2

# Dependências do Agente (LangChain 0.2.x e Gemini)
langchain~=0.2.0
langchain-core~=0.2.0
langchain-community~=0.0.38
langchain-google-genai~=1.0.0 # Conector oficial para Gemini
```

```bash
pip install -r requirements.txt
```

### 4. Execução do Aplicativo
Execute o Streamlit a partir da raiz do seu projeto:

```bash
streamlit run eda-agent-main.py 
```

O aplicativo será aberto no seu navegador. Siga os passos na barra lateral para carregar um arquivo CSV e fornecer sua chave de API para começar a interagir com o agente.

### 5. Melhorias Futuras
* **Melhoria na compreensão da linguagem:**
* **Tratamento de Erros mais claro:**
* * **Cache e Otimização de tokens:** 




