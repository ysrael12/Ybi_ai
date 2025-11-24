import streamlit as st
import pandas as pd
import random
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from mangaba import Agent  # Usando a estrutura de agentes do Mangaba

os.environ["GOOGLE_API_KEY"] = "AIzaSyDMhAkRxGY0cQHmFAU8zZJ51bijTxiuAr4"
os.environ["LLM_PROVIDER"] = "google" # Define um provider padrão para evitar erro
# Evita aviso de thread no Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="YBY.AI - Monitoramento IoT (CPU Mode)",
    page_icon="🥭",
    layout="wide"
)

# --- 1. CARREGAMENTO DO MODELO OTIMIZADO PARA WINDOWS/CPU ---
@st.cache_resource
def load_model_cpu():
    """
    Carrega o modelo base TinyLlama e os adaptadores de forma compatível com CPU/Windows.
    Evita bitsandbytes e foca em estabilidade.
    """
    # Identificadores do Modelo
    BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ADAPTER_REPO_ID = "YsraelJS/tinyllama-solo-management-adapters"
    
    status_container = st.empty()
    status_container.info("🖥️ Iniciando carregamento do modelo em modo CPU (Windows)...")
    
    try:
        # 1. Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # 2. Carregar Modelo Base (Forçando CPU e Float32 para compatibilidade máxima)
        status_container.info("⏳ Carregando modelo base na memória RAM (pode levar alguns minutos)...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="cpu",              # Força CPU
            torch_dtype=torch.float32,     # Float32 é o padrão mais seguro para CPU
            low_cpu_mem_usage=True         # Otimiza o carregamento na RAM
        )
        
        # 3. Carregar e Acoplar Adaptadores
        status_container.info("🔗 Baixando e aplicando adaptadores LoRA...")
        model = PeftModel.from_pretrained(
            base_model, 
            ADAPTER_REPO_ID,
            device_map="cpu"
        )
        
        # Merge para otimizar a inferência (remove a sobrecarga do LoRA na execução)
        model = model.merge_and_unload()
        
        status_container.success("✅ Modelo carregado e pronto para CPU!")
        return tokenizer, model
        
    except Exception as e:
        status_container.error(f"❌ Erro crítico ao carregar modelo: {e}")
        st.error("Dica: Verifique sua conexão com a internet e se tem pelo menos 4GB de RAM livres.")
        return None, None

# Carrega o modelo (Singleton via cache do Streamlit)
tokenizer, model = load_model_cpu()

# --- 2. ENGINE LOCAL PARA MANGABA AI ---
def run_mangaba_local(agent: Agent, user_input: str, tokenizer, model):
    """
    Executa a inferência localmente.
    """
    if not model:
        return "⚠️ Erro: Modelo offline ou não carregado."

    # Prompt Template (ChatML)
    system_message = f"Você é {agent.role}. {agent.backstory}. Seu objetivo é: {agent.goal}."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    
    # Prepara input
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cpu")  # Garante que os dados estejam na CPU
    
    # Geração
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=200,    # Reduzido levemente para ser mais rápido na CPU
            do_sample=True, 
            temperature=0.3,       # Baixa criatividade para ser mais técnico
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# --- 3. SIMULADOR IOT ---
st.sidebar.title("📡 IOT Server (Simulado)")
st.sidebar.caption("Rodando Localmente (Windows/CPU)")

if 'iot_data' not in st.session_state:
    st.session_state['iot_data'] = {
        'Temparature': 26.0, 'Humidity': 55.0, 'Moisture': 40.0,
        'Soil Type': 'Sandy', 'Crop Type': 'Maize',
        'Nitrogen': 20, 'Potassium': 10, 'Phosphorous': 10
    }

if st.sidebar.button("🔄 Atualizar Leituras"):
    st.session_state['iot_data'] = {
        'Temparature': round(random.uniform(22.0, 38.0), 1),
        'Humidity': round(random.uniform(40.0, 80.0), 1),
        'Moisture': round(random.uniform(20.0, 60.0), 1),
        'Soil Type': random.choice(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']),
        'Crop Type': random.choice(['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Wheat']),
        'Nitrogen': random.randint(5, 50),
        'Potassium': random.randint(5, 50),
        'Phosphorous': random.randint(5, 50)
    }

data = st.session_state['iot_data']

# Exibição Visual dos Dados
st.sidebar.markdown("---")
c1, c2 = st.sidebar.columns(2)
c1.metric("🌡️ Temp", f"{data['Temparature']}°C")
c2.metric("💧 Ar", f"{data['Humidity']}%")
c1.metric("🌱 Solo", f"{data['Moisture']}%")
st.sidebar.info(f"**Solo:** {data['Soil Type']}")
st.sidebar.warning(f"**Cultura:** {data['Crop Type']}")

st.sidebar.markdown("### Nutrientes (NPK)")
cn, cp, ck = st.sidebar.columns(3)
cn.metric("N", data['Nitrogen'])
cp.metric("P", data['Phosphorous'])
ck.metric("K", data['Potassium'])

# --- 4. INTERFACE PRINCIPAL ---

st.title("🥭 Yby AI: Manejo Inteligente")

# Tabs
tab1, tab2 = st.tabs(["📝 Relatório Técnico", "💬 Chatbot Agrônomo"])

# --- ABA 1: RELATÓRIO ---
with tab1:
    st.subheader("Análise de Solo em Tempo Real")
    st.markdown("O agente analisará os dados recebidos para recomendar a correção do solo.")
    
    st.dataframe(pd.DataFrame([data]), hide_index=True)

    if st.button("🚀 Analisar Solo e Gerar Relatório"):
        if not model:
            st.error("Aguarde o carregamento do modelo.")
        else:
            with st.spinner("Processando na CPU (isso pode levar alguns segundos)..."):
                
                # Agente Técnico
                agente_tecnico = Agent(
                    role="Especialista Agrônomo",
                    goal="Recomendar fertilizante baseado estritamente nos dados NPK e solo.",
                    backstory="Você é um sistema técnico preciso. Responda apenas com a recomendação fundamentada.",
                    verbose=True
                )

                # Prompt Estruturado (Igual ao Treinamento)
                prompt_input = (
                    f"Com uma temperatura de {data['Temparature']}, umidade de {data['Humidity']}, "
                    f"umidade do solo de {data['Moisture']}, e um solo do tipo {data['Soil Type']} "
                    f"para cultivar {data['Crop Type']}, e os níveis de nitrogênio, potássio e "
                    f"fósforo sendo {data['Nitrogen']}, {data['Potassium']}, {data['Phosphorous']}, "
                    f"qual é o fertilizante recomendado?"
                )

                res = run_mangaba_local(agente_tecnico, prompt_input, tokenizer, model)
                
                st.success("Análise Finalizada")
                st.info(f"**Recomendação:** {res}")

# --- ABA 2: CHATBOT ---
with tab2:
    st.subheader("Consultor Virtual")
    
    agente_chat = Agent(
        role="Assistente Rural",
        goal="Responder dúvidas gerais de forma curta e simples.",
        backstory="Assistente virtual amigável.",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_query := st.chat_input("Dúvidas? (Ex: Como aplicar Ureia?)"):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.chat_message("assistant"):
            if model:
                with st.spinner("Escrevendo..."):
                    resp = run_mangaba_local(agente_chat, user_query, tokenizer, model)
                    st.write(resp)
                    st.session_state.chat_history.append({"role": "assistant", "content": resp})
            else:
                st.error("Modelo ainda carregando...")