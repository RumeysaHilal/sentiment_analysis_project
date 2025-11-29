import numpy as np

# --- Parametreler ve Embeddingler ---
SEQ_LEN = 5
D_MODEL = 4
E_embeddings = np.array([
    [0.5, 0.2, -0.3, 0.8], 
    [0.1, 0.6, 0.7, -0.2], 
    [-0.4, 0.3, 0.9, 0.5], 
    [0.9, -0.1, 0.2, 0.6], 
    [0.2, 0.7, -0.5, 0.4]  
])

# --- RFF PE Fonksiyonu ---
def rff_pe_numpy(seq_len, d_model, scale=0.5):
    # Rastgele Frekans (W) ve Faz (b) matrisleri oluşturulur
    W = np.random.normal(loc=0.0, scale=scale, size=(d_model, 1))
    b = np.random.uniform(low=0.0, high=2 * np.pi, size=(d_model, 1))
    
    t_positions = np.arange(1, seq_len + 1)
    
    # Argüman hesabı: W * t + b
    # W @ t_positions[None, :] işlemi matris çarpımıyla (d_model, seq_len) boyutunu verir
    arg = W @ t_positions[None, :] + b 
    
    # Kosinüs uygulanır ve transpoze alınır: P_t (seq_len, d_model)
    P_t = np.cos(arg).T 
    
    return P_t

# --- Uygulama ---
P_rff = rff_pe_numpy(SEQ_LEN, D_MODEL)
E_final_rff = E_embeddings + P_rff

print("--- Nihai Konumsal Kodlanmış Embeddingler (RFF PE) ---")
for i, kelime in enumerate(["Doğal", "dil", "işleme", "ödevini", "yaptım."]):
    print(f"{kelime:<8}: {E_final_rff[i].round(decimals=3)}")