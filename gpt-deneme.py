import torch
import math

# 1️⃣ Cümle (tokenlar)
tokens = ["doğal", "dil", "işleme", "ödevini", "yaptım"]

# 2️⃣ Sahte embedding vektörleri (örnek)
embeddings = torch.tensor([
    [0.1, 0.3, 0.7],
    [0.2, 0.4, 0.6],
    [0.3, 0.5, 0.5],
    [0.4, 0.5, 0.3],
    [0.5, 0.6, 0.2]
])

# 3️⃣ Fourier PE fonksiyonu
def fourier_encode_positions(num_positions, num_bands=4, max_freq=5.0):
    positions = torch.linspace(0, 1, num_positions).unsqueeze(1)
    freq_bands = torch.linspace(1.0, max_freq, num_bands)
    angles = 2 * math.pi * positions * freq_bands
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return pe  # [num_positions, num_bands*2]

# 4️⃣ Pozisyon kodlarını al
pe = fourier_encode_positions(len(tokens))

# 5️⃣ Birleştir
encoded = torch.cat([embeddings, pe], dim=1)

print("Sonuç boyutu:", encoded.shape)
print(encoded)
