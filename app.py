import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Streamlit temel ayar
# -----------------------------
st.set_page_config(page_title="Gradient Descent Lab", page_icon="⬇️")

st.title("⬇️ Gradient Descent Simülatörü – Yokuştan Aşağı İnen Top")
st.write(
    """
Bu laboratuvarda tek değişkenli bir fonksiyon üzerinde
**gradient descent (eğim azalışı)** yöntemini inceleyeceksin.

- Bir fonksiyon seç (parabol veya çukurlu fonksiyon)  
- Başlangıç noktasını (**x₀**) belirle  
- Öğrenme oranını (**η**) ve adım sayısını seç  
- Topun her adımda vadinin tabanına nasıl yaklaştığını grafikte izle
"""
)

st.markdown("---")


# -----------------------------
# Fonksiyon seçimi
# -----------------------------
st.subheader("1️⃣ Fonksiyonu Seç")

func_name = st.radio(
    "Fonksiyon:",
    [
        "Basit Parabol: f(x) = x²",
        "Çukurlu Fonksiyon: f(x) = x⁴/4 − x²/2",
    ],
)


def f(x: np.ndarray, name: str) -> np.ndarray:
    """Seçilen fonksiyonun değeri."""
    if name == "Basit Parabol: f(x) = x²":
        return x**2
    elif name == "Çukurlu Fonksiyon: f(x) = x⁴/4 − x²/2":
        return (x**4) / 4 - (x**2) / 2
    else:
        return x**2


def f_prime(x: np.ndarray, name: str) -> np.ndarray:
    """Seçilen fonksiyonun türevi."""
    if name == "Basit Parabol: f(x) = x²":
        return 2 * x
    elif name == "Çukurlu Fonksiyon: f(x) = x⁴/4 − x²/2":
        # f'(x) = x³ - x
        return x**3 - x
    else:
        return 2 * x


# Grafiği çizmek için x aralığı
if func_name == "Basit Parabol: f(x) = x²":
    x_min, x_max = -5.0, 5.0
else:
    x_min, x_max = -3.0, 3.0

x_plot = np.linspace(x_min, x_max, 400)
y_plot = f(x_plot, func_name)


# -----------------------------
# Gradient descent parametreleri
# -----------------------------
st.subheader("2️⃣ Gradient Descent Parametrelerini Ayarla")

col_params1, col_params2 = st.columns(2)
with col_params1:
    x0 = st.slider(
        "Başlangıç noktası x₀",
        min_value=float(x_min),
        max_value=float(x_max),
        value=2.5 if func_name == "Basit Parabol: f(x) = x²" else 2.0,
        step=0.1,
    )

with col_params2:
    eta = st.slider(
        "Öğrenme oranı (η)",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Adım boyu. Çok küçük olursa yavaş, çok büyük olursa zıplayarak sapıtabilir.",
    )

n_steps = st.slider(
    "Adım sayısı",
    min_value=1,
    max_value=50,
    value=15,
    step=1,
)

st.write(
    f"Başlangıç: **x₀ = {x0:.2f}**, öğrenme oranı: **η = {eta:.2f}**, adım sayısı: **{n_steps}**"
)

st.markdown(
    """
Gradient descent adım formülü:

\\[
x_{k+1} = x_k - \\eta \\, f'(x_k)
\\]

Burada \\(f'(x_k)\\) fonksiyonun o noktadaki eğimidir (türev).
"""
)


# -----------------------------
# Gradient descent adımlarını hesapla
# -----------------------------
xs = [x0]
ys = [f(np.array([x0]), func_name)[0]]

x_curr = x0
for _ in range(n_steps):
    grad = f_prime(np.array([x_curr]), func_name)[0]
    x_next = x_curr - eta * grad
    xs.append(x_next)
    ys.append(f(np.array([x_next]), func_name)[0])
    x_curr = x_next

xs = np.array(xs)
ys = np.array(ys)


# -----------------------------
# Görselleştirme
# -----------------------------
st.markdown("---")
st.subheader("3️⃣ Grafikte Gradient Descent Adımlarını İncele")

fig, ax = plt.subplots(figsize=(7, 5))

# Fonksiyon eğrisi
ax.plot(x_plot, y_plot, label="f(x)")

# Adım noktaları
ax.scatter(xs, ys, label="Adımlar (x_k)", zorder=3)
ax.plot(xs, ys, linestyle="--", alpha=0.7)

# İlk ve son noktayı etiketle
ax.scatter(xs[0], ys[0], s=60)
ax.text(xs[0], ys[0], "  Başlangıç", va="bottom")

ax.scatter(xs[-1], ys[-1], s=60)
ax.text(xs[-1], ys[-1], "  Son", va="bottom")

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Gradient Descent Yolu")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend()

st.pyplot(fig)


# -----------------------------
# Adım tablosu
# -----------------------------
st.subheader("4️⃣ Adım Adım Sayısal Sonuçlar")

import pandas as pd  # Streamlit için tablo

step_indices = np.arange(len(xs))
grads = f_prime(xs, func_name)
df_steps = pd.DataFrame(
    {
        "k (adım)": step_indices,
        "x_k": xs,
        "f(x_k)": ys,
        "f'(x_k)": grads,
    }
)

st.dataframe(df_steps.style.format({"x_k": "{:.4f}", "f(x_k)": "{:.4f}", "f'(x_k)": "{:.4f}"}))


# -----------------------------
# Açıklama / Öğretmen kutusu
# -----------------------------
st.markdown("---")
st.info(
    "Gradient descent, fonksiyonun türevine bakarak her adımda "
    "değerimizi en hızlı azalış yönünde güncelleyen basit ama güçlü bir optimizasyon yöntemidir. "
    "Yeterince küçük bir öğrenme oranı ile, uygun başlangıç noktalarından minimuma doğru yaklaşırız."
)

with st.expander("👩‍🏫 Öğretmen Kutusu – 1D Gradient Descent Sezgisi"):
    st.write(
        r"""
Tek değişkenli bir fonksiyon için gradient descent adımı:

\\[
x_{k+1} = x_k - \eta \, f'(x_k)
\\]

- Eğer \\(f'(x_k) > 0\\) ise, fonksiyon sağa doğru **yükseliyor** demektir → sola gitmek isteriz.  
  Bu nedenle \\(- \eta f'(x_k)\\) negatiftir → \\(x_{k+1} < x_k\\).
- Eğer \\(f'(x_k) < 0\\) ise, fonksiyon sola doğru **yükseliyor** demektir → sağa gitmek isteriz.  
  Bu nedenle \\(- \eta f'(x_k)\\) pozitiftir → \\(x_{k+1} > x_k\\).

Öğrenme oranı \\(\eta\\):

- Çok küçük → yavaş ilerleme, ama genelde daha kararlı.  
- Çok büyük → minimumu atlayıp sağa–sola zıplayabilir, bazen diverge olabilir.

Bu labda öğrenciler:

1. Farklı \\(x_0\\) ve \\(\eta\\) seçimlerinin yola etkisini görsel olarak inceler,  
2. Aynı fonksiyonun farklı başlangıçlardan nasıl farklı yollarla ama benzer minima'lara gittiğini gözlemler.
"""
    )

st.caption(
    "Bu modül, lise sonu / üniversite başı düzeyinde türev ve optimizasyon kavramlarına "
    "sezgisel bir giriş sağlamak için tasarlanmıştır."
)
