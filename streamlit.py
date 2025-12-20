import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

# ==============================================================================
# 🧠 1. EXACT MODEL ARCHITECTURE (From Notebook)
# ==============================================================================

class EnhancedHormoneAttentionHead(nn.Module):
    def __init__(self, hidden_dim: int, hormone_name: str, hormone_idx: int, 
                 num_heads: int = 4, temperature: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hormone_name = hormone_name
        self.hormone_idx = hormone_idx
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        
        # Learnable hormone query
        self.hormone_query = nn.Parameter(torch.zeros(1, num_heads, self.head_dim))
        # Note: _init_orthogonal_query is skipped here as we load trained weights
        
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attended_norm = nn.LayerNorm(hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        self.last_attention_weights = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        keys = self.key_proj(hidden_states)
        values = self.value_proj(hidden_states)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = self.hormone_query.expand(batch_size, -1, -1).unsqueeze(2)
        scale = math.sqrt(self.head_dim) * self.temperature
        scores = torch.matmul(query, keys.transpose(-2, -1)) / scale
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        self.last_attention_weights = attention_weights.detach()
        attended = torch.matmul(attention_weights, values)
        attended = attended.squeeze(2).view(batch_size, self.hidden_dim)
        attended = self.attended_norm(attended)
        output = self.output_proj(attended)
        return torch.sigmoid(output + self.output_bias)

class HormoneEmotionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_hormones: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hormones = num_hormones
        self.hormone_names = ["dopamine", "serotonin", "cortisol", "oxytocin", "adrenaline", "endorphins"]
        
        self.hormone_heads = nn.ModuleDict({
            name: EnhancedHormoneAttentionHead(hidden_dim, name, idx)
            for idx, name in enumerate(self.hormone_names)
        })
        
        self.hormone_to_embedding = nn.Sequential(
            nn.Linear(num_hormones, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.modulation_strength = nn.Parameter(torch.tensor(0.2))
        self._inference_activations = None

    def forward(self, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        hormone_values = []
        for name in self.hormone_names:
            value = self.hormone_heads[name](encoder_hidden_states, attention_mask)
            hormone_values.append(value)
        
        hormones = torch.cat(hormone_values, dim=-1)
        # Store for visualization
        self._inference_activations = hormones.detach()
        
        emotional_embedding = self.hormone_to_embedding(hormones)
        emotional_expanded = emotional_embedding.unsqueeze(1)
        strength = self.modulation_strength.clamp(0.1, 0.5)
        return encoder_hidden_states * (1.0 + strength * emotional_expanded)
    
    def get_hormone_activations(self) -> dict:
        if self._inference_activations is None:
            return {h: 0.5 for h in self.hormone_names}
        acts = self._inference_activations[0].cpu().numpy()
        return {self.hormone_names[i]: float(acts[i]) for i in range(len(self.hormone_names))}

class HormoneT5(nn.Module):
    def __init__(self, model_name: str = "t5-small", freeze_backbone: bool = True):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config = self.t5.config
        self.hormone_block = HormoneEmotionBlock(self.config.d_model)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        modified_hidden = self.hormone_block(encoder_outputs.last_hidden_state, attention_mask)
        modified_encoder_outputs = BaseModelOutput(
            last_hidden_state=modified_hidden,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )
        return self.t5(encoder_outputs=modified_encoder_outputs, attention_mask=attention_mask, labels=labels, return_dict=True)
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        # 1. Run Encoder
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # 2. Modulate with Hormones (This updates the hormone values!)
        modified_hidden = self.hormone_block(encoder_outputs.last_hidden_state, attention_mask)
        # 3. Package for Decoder
        modified_encoder_outputs = BaseModelOutput(last_hidden_state=modified_hidden)
        # 4. Generate
        return self.t5.generate(encoder_outputs=modified_encoder_outputs, attention_mask=attention_mask, **kwargs)

    def encode_only(self, input_ids, attention_mask=None):
        """Helper to get hormones without generating text."""
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        _ = self.hormone_block(encoder_outputs.last_hidden_state, attention_mask)
        return self.hormone_block.get_hormone_activations()

# ==============================================================================
# 🛠️ 2. STREAMLIT APP CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Hormone-Based Emotional AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for a cleaner look
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    .hormone-title { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .info-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #cce5ff;
        font-size: 14px;
    }
    .stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORMONES = ["dopamine", "serotonin", "cortisol", "oxytocin", "adrenaline", "endorphins"]
# Colors: Green, Blue, Red, Pink, Orange, Yellow
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#fd79a8', '#f39c12', '#f1c40f']

# ==============================================================================
# 📂 3. LOAD MODEL
# ==============================================================================

@st.cache_resource
def load_emotional_model():
    """Load model weights and tokenizer."""
    try:
        # Expected paths
        model_path = "model/hormone_t5_weights.pth"
        tokenizer_path = "model/tokenizer"
        
        # Load Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        
        # Initialize Model
        model = HormoneT5("t5-small", freeze_backbone=True)
        
        # Load Weights
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.warning("Make sure you have a 'model' folder with 'hormone_t5_weights.pth' and 'tokenizer' folder inside.")
        return None, None

model, tokenizer = load_emotional_model()

# ==============================================================================
# 📊 4. VISUALIZATION & LOGIC
# ==============================================================================

def get_emotional_state(hormones):
    """Determine emotional state (Logic matches Notebook exactly)."""
    dop = hormones.get("dopamine", 0.5)
    ser = hormones.get("serotonin", 0.5)
    cort = hormones.get("cortisol", 0.5)
    oxy = hormones.get("oxytocin", 0.5)
    adren = hormones.get("adrenaline", 0.5)
    
    if cort > 0.7 and dop < 0.3:
        return "😤 STRESSED/ANGRY", "I'm feeling stressed and upset!"
    elif dop > 0.7 and adren > 0.7 and cort < 0.3:
        return "🤩 EXCITED", "I'm so excited and energized!"
    elif dop > 0.7 and ser > 0.7 and cort < 0.3:
        return "😊 HAPPY", "I'm feeling great and positive!"
    elif oxy > 0.7 and dop < 0.3:
        return "🥺 SAD/EMPATHETIC", "I'm feeling sad but connected..."
    elif all(0.3 < h < 0.7 for h in [dop, ser, cort]):
        return "😐 NEUTRAL", "I'm feeling balanced and calm."
    else:
        return "🤔 MIXED/COMPLEX", "I'm experiencing complex emotions..."

def draw_hormone_chart(hormone_data):
    """Draw bar chart using Matplotlib."""
    if not hormone_data:
        hormone_data = {h: 0.5 for h in HORMONES}
    
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    x = np.arange(len(HORMONES))
    values = [hormone_data[h] for h in HORMONES]
    
    bars = ax.bar(x, values, color=COLORS, alpha=0.9, edgecolor='black', width=0.6)
    
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([h.upper()[:4] for h in HORMONES], fontweight='bold')
    ax.set_yticks([]) # Hide y-axis numbers
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    return fig

# ==============================================================================
# 🖥️ 5. APP UI
# ==============================================================================

st.title("🧠 Hormone-Based Emotional AI")

# Init Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hormones" not in st.session_state:
    st.session_state.hormones = {h: 0.5 for h in HORMONES}

# --- TOP SECTION: VISUALIZATION ---
top_col1, top_col2 = st.columns([3, 1])

with top_col1:
    st.markdown("##### 🧬 Live Hormone Levels")
    chart_placeholder = st.empty()
    fig = draw_hormone_chart(st.session_state.hormones)
    try:
        chart_placeholder.pyplot(fig, use_container_width=True)
    except:
        chart_placeholder.pyplot(fig)
    plt.close(fig)

with top_col2:
    state_label, feeling_desc = get_emotional_state(st.session_state.hormones)
    st.markdown(f"**State:** {state_label}")
    st.markdown("""
        <div class="info-box">
        <b>Key:</b><br>
        🟢 <b>Dopamine:</b> Reward/Joy<br>
        🔵 <b>Serotonin:</b> Calm/Well-being<br>
        🔴 <b>Cortisol:</b> Stress/Defense<br>
        💗 <b>Oxytocin:</b> Love/Empathy<br>
        🟠 <b>Adrenaline:</b> Energy/Shock<br>
        🟡 <b>Endorphins:</b> Relief/Euphoria
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- CHAT AREA ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT HANDLING ---
if prompt := st.chat_input("Say something (e.g., 'I hate you', 'I won!', 'I'm sad')..."):
    
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. AI Processing
    if model and tokenizer:
        with st.chat_message("assistant"):
            msg_container = st.empty()
            msg_container.markdown("🧠 *Thinking & Feeling...*")
            
            # Prepare Input
            input_text = f"{prompt}"
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            ).to(DEVICE)
            
            with torch.no_grad():
                # A. GENERATE (Matches Notebook Chat Function Parameters Exactly)
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=80,
                    num_beams=4,          # Notebook used 4
                    early_stopping=True,
                    no_repeat_ngram_size=3, # Notebook used 3
                    do_sample=True,
                    temperature=0.8,      # Notebook used 0.8
                    top_p=0.92            # Notebook used 0.92
                )
                
                response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # B. UPDATE HORMONES
                # Note: generate() calls forward() which updates the hormones internally
                # We can just fetch them now.
                new_hormones = model.hormone_block.get_hormone_activations()
            
            # 3. Update UI
            msg_container.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.hormones = new_hormones
            st.rerun() # Rerun to update the top chart immediately