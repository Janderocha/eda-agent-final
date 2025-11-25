# Agente Autônomo de EDA (Exploratory Data Analysis)

## 📋 Descrição da Solução

Este projeto implementa um agente autônomo capaz de realizar análise exploratória de dados em qualquer arquivo CSV. O agente utiliza LangChain para orquestração, OpenAI GPT-4 como motor de raciocínio, e Streamlit para interface web interativa.

## 🏗️ Arquitetura

### Framework Escolhida
- **LangChain**: Orquestração de agentes e memória
- **OpenAI GPT-4**: Motor de LLM para raciocínio
- **Streamlit**: Interface web interativa
- **Pandas/NumPy**: Processamento de dados
- **Matplotlib/Seaborn/Plotly**: Visualizações

### Estrutura da Solução

```
┌─────────────────────────────────┐
│     Interface Streamlit         │
│  (Upload CSV + Chat Interface)  │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│   LangChain Agent Executor      │
│   (OpenAI Functions Agent)      │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼─────┐ ┌────▼──────┐
│ Analysis  │ │ Memory    │
│ Tools     │ │ System    │
│ (8 tools) │ │           │
└───────────┘ └───────────┘
```

### Ferramentas Implementadas

1. **describe_data**: Estatísticas descritivas completas
2. **check_missing**: Análise de valores faltantes
3. **calculate_correlation**: Matriz de correlação
4. **detect_outliers**: Detecção de outliers (IQR e Z-score)
5. **generate_plot**: Geração de gráficos (5 tipos)
6. **get_columns**: Lista de colunas disponíveis
7. **save_conclusion**: Salva insights importantes
8. **get_conclusions**: Recupera conclusões salvas

## 🚀 Instalação

### Pré-requisitos
- Python 3.9+
- Pip

### Passo a Passo

1. **Clone ou baixe os arquivos**

2. **Crie um ambiente virtual** (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Execute a aplicação**
```bash
streamlit run eda_agent.py
```

5. **Acesse no navegador**
```
http://localhost:8501
```

## 📖 Como Usar

### 1. Configuração Inicial
- Insira sua **OpenAI API Key** na barra lateral
- Faça **upload de um arquivo CSV**
- Aguarde confirmação do carregamento

### 2. Fazendo Perguntas

O agente entende perguntas em linguagem natural. Exemplos:

#### Análise Descritiva
```
"Descreva os dados deste arquivo"
"Quais são as estatísticas básicas das variáveis numéricas?"
"Mostre informações sobre valores faltantes"
```

#### Visualizações
```
"Crie um histograma da coluna Amount"
"Mostre um gráfico de dispersão entre Time e Amount"
"Gere uma matriz de correlação"
"Faça um boxplot das variáveis V1, V2 e V3"
```

#### Análise de Padrões
```
"Existem outliers na coluna Amount?"
"Quais variáveis estão mais correlacionadas?"
"Qual a distribuição da variável Class?"
```

#### Conclusões
```
"Quais conclusões você obteve até agora?"
"Resuma os principais insights desta análise"
"O que você descobriu sobre fraudes neste dataset?"
```

### 3. Sistema de Memória

O agente automaticamente:
- Salva conclusões importantes durante análises
- Mantém contexto da conversa
- Pode recuperar insights anteriores quando solicitado

## 📊 Exemplos de Uso com Credit Card Fraud

### Pergunta 1: Análise Descritiva
```
Usuário: "Descreva os dados deste arquivo. Quantas transações temos e qual a proporção de fraudes?"

Agente irá:
1. Usar get_columns para ver as colunas
2. Usar describe_data para estatísticas
3. Calcular proporção de fraudes
4. Salvar conclusão sobre desbalanceamento
```

### Pergunta 2: Visualização de Distribuição
```
Usuário: "Crie um histograma da coluna Amount e me diga o que observa"

Agente irá:
1. Gerar histograma com generate_plot
2. Analisar a distribuição
3. Salvar conclusão sobre padrão observado
```

### Pergunta 3: Análise de Correlação
```
Usuário: "Quais variáveis têm maior correlação com fraudes?"

Agente irá:
1. Calcular matriz de correlação
2. Identificar maiores correlações com Class
3. Gerar heatmap de correlação
4. Salvar insights sobre variáveis importantes
```

### Pergunta 4: Detecção de Anomalias
```
Usuário: "Existem outliers na coluna Amount? Como isso se relaciona com fraudes?"

Agente irá:
1. Detectar outliers usando IQR
2. Analisar relação com Class
3. Salvar conclusões sobre padrões de fraude
```

### Pergunta 5: Síntese Final
```
Usuário: "Quais são suas conclusões sobre este dataset de fraudes?"

Agente irá:
1. Usar get_conclusions para recuperar insights
2. Sintetizar descobertas principais
3. Fornecer recomendações
```

## 🔧 Personalização

### Adicionar Novas Ferramentas

```python
def tool_custom_analysis(query: str) -> str:
    """Sua análise customizada"""
    # Seu código aqui
    return resultado

# Adicione à lista de tools
tools.append(
    Tool(
        name="custom_analysis",
        func=tool_custom_analysis,
        description="Descrição para o agente"
    )
)
```

### Modificar Tipos de Gráficos

Edite a função `generate_plot()` na classe `DataAnalyzer` para adicionar novos tipos.

### Ajustar Prompt do Sistema

Modifique a variável `system_prompt` para alterar o comportamento do agente.

## 📦 Deploy

### Streamlit Cloud (Gratuito)

1. Crie conta no [Streamlit Cloud](https://streamlit.io/cloud)
2. Conecte seu repositório GitHub
3. Configure secrets (API keys) nas configurações
4. Deploy automático!

### Docker (Opcional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "eda_agent.py"]
```

## 🔐 Segurança

- ⚠️ **NUNCA** commite API keys no código
- Use variáveis de ambiente ou Streamlit secrets
- No Streamlit Cloud: Settings → Secrets

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

```python
# No código
import streamlit as st
api_key = st.secrets.get("OPENAI_API_KEY", "")
```

## 📈 Capacidades do Agente

### ✅ Pode Responder

- [x] Tipos de dados (numéricos, categóricos)
- [x] Distribuição de variáveis (histogramas)
- [x] Intervalos (mínimo, máximo)
- [x] Tendência central (média, mediana)
- [x] Variabilidade (desvio padrão, variância)
- [x] Padrões temporais
- [x] Valores frequentes/raros
- [x] Detecção de outliers
- [x] Correlações entre variáveis
- [x] Gráficos variados
- [x] **Conclusões baseadas nas análises**

### 🎨 Tipos de Gráficos

1. **Histogram**: Distribuição de uma variável
2. **Scatter**: Relação entre duas variáveis
3. **Boxplot**: Distribuição e outliers
4. **Correlation**: Heatmap de correlação
5. **Bar**: Frequência de valores categóricos

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError"
```bash
pip install -r requirements.txt --upgrade
```

### Erro: "API key invalid"
- Verifique se a chave está correta
- Confirme que tem créditos na conta OpenAI

### Gráficos não aparecem
- Verifique se o CSV foi carregado corretamente
- Confirme que as colunas especificadas existem

### Agente não responde adequadamente
- Aumente `max_iterations` no AgentExecutor
- Verifique logs do Streamlit (terminal)
- Melhore a descrição das tools

## 📝 Estrutura de Arquivos para Entrega

```
📁 Agentes_Autonomos_EDA/
├── 📄 eda_agent.py (código principal)
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 Agentes Autônomos – Relatório da Atividade Extra.pdf
├── 📄 .gitignore
└── 📁 exemplos/
    ├── exemplo_fraudes.png
    ├── exemplo_correlacao.png
    └── exemplo_conclusoes.png
```

## 🎯 Diferencias Competitivos

1. **Interface Intuitiva**: Chat natural, sem comandos complexos
2. **Memória Persistente**: Salva conclusões automaticamente
3. **Versatilidade**: Funciona com qualquer CSV
4. **Visualizações Automáticas**: Gera gráficos apropriados
5. **Análise Completa**: 8 ferramentas especializadas
6. **Escalável**: Fácil adicionar novas capacidades

## 📚 Referências

- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)

## 👥 Suporte

Para dúvidas sobre este projeto:
- Consulte a documentação das bibliotecas
- Revise os exemplos fornecidos
- Experimente perguntas variadas ao agente

---

**Desenvolvido para**: Atividade Obrigatória - Agentes Autônomos
**Instituição**: Institut d'Intelligence Artificielle Appliquée
**Data**: 2025
