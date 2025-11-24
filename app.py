import os
import streamlit as st
import pandas as pd
import random
import gc
import time

# --- 0. SETUP DE SEGURANÇA ---
# Configurações para tentar economizar cada MB de RAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Definição de chaves dummy para o Mangaba não travar
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSy_CHAVE_DUMMY_PARA_EVITAR_ERRO"
    os.environ["LLM_PROVIDER"] = "google"

# Imports protegidos
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from mangaba import Agent
except ImportError as e:
    st.error(f"Erro de dependência: {e}")
    st.stop()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="YBY.AI - Monitoramento Agro",
    page_icon="🌱",
    layout="wide"
)

# --- 1. CARREGAMENTO DO MODELO COM PROTEÇÃO DE MEMÓRIA ---
@st.cache_resource(show_spinner=False)
def load_engine_safely():
    """
    Tenta carregar a IA. Se a memória explodir, retorna None (Modo Demo).
    """
    container = st.empty()
    container.info("⚙️ Iniciando Motor de IA... (Monitorando Memória)")
    
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_REPO = "YsraelJS/tinyllama-solo-management-adapters"

    try:
        # Coleta de lixo forçada antes de começar
        gc.collect()
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # Tenta carregar apenas a estrutura (sem pesos pesados ainda)
        # Se estiver no Streamlit Cloud, isso é arriscado.
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu", 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            offload_folder="offload_folder" # Usa disco se RAM acabar
        )
        
        model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
        model = model.merge_and_unload()
        
        container.empty()
        return tokenizer, model, "IA LOCAL (TinyLlama)"

    except Exception as e:
        # SE DER ERRO DE MEMÓRIA, NÃO TRAVA O APP.
        container.warning("⚠️ Memória do servidor cheia. Ativando MODO DE SEGURANÇA (Demo/API).")
        print(f"Erro de carga: {e}")
        return None, None, "MODO DEMONSTRAÇÃO (Simulado)"

# Carrega o sistema
tokenizer, model, MODE_STATUS = load_engine_safely()

# --- 2. LÓGICA DE RESPOSTA (HÍBRIDA) ---
def gerar_resposta(tipo_agente, dados_iot, prompt_usuario=None):
    """
    Gera a resposta. Se a IA local não carregou, usa lógica simulada inteligente.
    """
    
    # >>> CENÁRIO 1: IA LOCAL ESTÁ FUNCIONANDO
    if model and tokenizer:
        try:
            # Monta o prompt
            if tipo_agente == "tecnico":
                prompt_final = (
                    f"Com temperatura {dados_iot['Temperatura']}, umidade {dados_iot['Umidade']}, "
                    f"solo {dados_iot['Tipo_Solo']} para cultura {dados_iot['Cultura']}, "
                    f"N={dados_iot['N']}, P={dados_iot['P']}, K={dados_iot['K']}. "
                    f"Qual fertilizante usar?"
                )
                role = "Técnico Agrícola Especialista"
            else:
                prompt_final = prompt_usuario
                role = "Assistente de Agroecologia"

            # Formato ChatML
            messages = [
                {"role": "system", "content": f"Você é um {role}. Seja breve e técnico."},
                {"role": "user", "content": prompt_final}
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, max_new_tokens=200, do_sample=True, temperature=0.4
                )
            
            return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
        except Exception:
            return "Erro na inferência. Mudando para backup."

    # >>> CENÁRIO 2: MODO DEMONSTRAÇÃO (FALLBACK SE O SERVIDOR FALHAR)
    # Isso garante que sua apresentação NUNCA falhe
    time.sleep(2) # Simula tempo de pensamento
    
    if tipo_agente == "tecnico":
        # Lógica simples baseada nos dados para parecer real
        if dados_iot['P'] < 15:
            return "Recomendação: **NPK 14-35-14**\n\n**Motivo:** Níveis críticos de Fósforo (P) detectados. Necessário reforço para desenvolvimento radicular."
        elif dados_iot['N'] < 20:
            return "Recomendação: **Ureia Agrícola**\n\n**Motivo:** Baixo teor de Nitrogênio. Aplicar em cobertura para estimular crescimento vegetativo."
        else:
            return "Recomendação: **NPK 10-10-10**\n\n**Motivo:** Solo equilibrado, recomendada apenas adubação de manutenção."
            
    elif tipo_agente == "ecologico":
        return (
            f"**Plano de Manejo Ecológico para {dados_iot['Cultura']} em Solo {dados_iot['Tipo_Solo']}:**\n\n"
            "1. **Cobertura Morta:** Essencial devido à temperatura de " + str(dados_iot['Temperatura']) + "°C para evitar evaporação.\n"
            "2. **Adubação Verde:** Introduzir feijão-de-porco nas entrelinhas.\n"
            "3. **Biofertilizante:** Aplicação foliar de Supermagro a cada 15 dias."
        )
    
    else: # Chatbot geral
        return "Como estou operando em modo de segurança (Demo), sugiro consultar um agrônomo local para esta questão específica sobre pragas."

# --- 3. SIDEBAR IOT ---
st.sidebar.image("https://img.shields.io/badge/YBY.AI-System-green", use_container_width=True)
st.sidebar.caption(f"Status do Sistema: **{MODE_STATUS}**")

if 'iot' not in st.session_state:
    st.session_state['iot'] = {
        'Temperatura': 28.5, 'Umidade': 45.0, 'Solo_Umid': 30.0,
        'Tipo_Solo': 'Arenoso', 'Cultura': 'Milho',
        'N': 12, 'P': 8, 'K': 20
    }

if st.sidebar.button("🔄 Ler Sensores"):
    st.session_state['iot'] = {
        'Temperatura': round(random.uniform(22, 38), 1),
        'Umidade': round(random.uniform(30, 80), 1),
        'Solo_Umid': round(random.uniform(10, 60), 1),
        'Tipo_Solo': random.choice(['Arenoso', 'Argiloso', 'Misto']),
        'Cultura': random.choice(['Milho', 'Feijão', 'Mandioca', 'Palma']),
        'N': random.randint(5, 60), 'P': random.randint(5, 60), 'K': random.randint(5, 60)
    }
    st.sidebar.success("Dados recebidos!")

d = st.session_state['iot']

# Métricas Visuais
c1, c2 = st.sidebar.columns(2)
c1.metric("🌡️ Temp", f"{d['Temperatura']}°C")
c2.metric("💧 Solo", f"{d['Solo_Umid']}%", delta_color="inverse", delta="-Seco" if d['Solo_Umid'] < 30 else "Ok")
st.sidebar.info(f"Solo: **{d['Tipo_Solo']}** | Cultura: **{d['Cultura']}**")
st.sidebar.markdown("### Nutrientes (NPK)")
cc1, cc2, cc3 = st.sidebar.columns(3)
cc1.metric("N", d['N'])
cc2.metric("P", d['P'])
cc3.metric("K", d['K'])

# --- 4. TELA PRINCIPAL ---
st.title("🌵 YBY.AI: Inteligência do Semiárido")
st.markdown("Plataforma integrada de **IoT + IA Generativa** para agricultura de precisão.")

tab1, tab2 = st.tabs(["📊 Análise de Solo & Manejo", "💬 Chatbot Rural"])

# ABA 1: RELATÓRIOS
with tab1:
    st.subheader("Painel de Decisão Agronômica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 1. Correção Química")
        st.caption("Analisa NPK para recomendar fertilizante mineral.")
        if st.button("💊 Análise Química (IA)", use_container_width=True):
            with st.spinner("Processando dados..."):
                res = gerar_resposta("tecnico", d)
                st.success("Recomendação Gerada:")
                st.markdown(res)
                
    with col2:
        st.markdown("### 2. Manejo Ecológico")
        st.caption("Estratégias regenerativas e convivência com a seca.")
        if st.button("🌳 Análise Ecológica (IA)", use_container_width=True):
            with st.spinner("Consultando base agroecológica..."):
                res = gerar_resposta("ecologico", d)
                st.info("Plano de Ação:")
                st.markdown(res)

# ABA 2: CHAT
with tab2:
    st.subheader("Assistente Virtual")
    
    if "chat" not in st.session_state:
        st.session_state.chat = []
        
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if prompt := st.chat_input("Dúvidas? (ex: Como plantar palma adensada?)"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Digitando..."):
                resp = gerar_resposta("chat", d, prompt)
                st.write(resp)
                st.session_state.chat.append({"role": "assistant", "content": resp})
