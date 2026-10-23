import numpy as np
import matplotlib.pyplot as plt


N = 100_000
SIGMA = 0.216 # cm^{-1}
A = 0.1 #cm

rng = np.random.default_rng(42)

xi = rng.random(N)
free_path = -np.log(xi) / SIGMA
collided = free_path < A
second_half = (free_path >= A / 2) & (free_path < A)

p_mc = collided.mean()
p_second_mc = second_half.mean()

p_exact = 1 - np.exp(-SIGMA * A)
p_second_exact = np.exp(-SIGMA * A / 2) - np.exp(-SIGMA * A)

p_error = np.sqrt(p_mc * (1 - p_mc) / N)

difference = abs(p_mc - p_exact)

print(f"Количество нейтронов: {N}")
print(f"Столкновений внутри мишени: {collided.sum()}")
print(f"Вероятность (Монте-Карло): {p_mc:.6f}")
print(f"Вероятность (аналитическая): {p_exact:.6f}")
print()
print(f"Столкновений во второй половине: {second_half.sum()}")
print(f"Вероятность столкновения во второй половине (Монте-Карло): {p_second_mc:.6f}")
print(f"Вероятность столкновений во второй половине (аналитическая): {p_second_exact:.6f}")
print()
print(f"Статистическая ошибка: {p_error:.6f}")
print(f"Абсолютное отклонение: {difference:.6f}")
print(f"Отклонение в единицах ошибки: {difference / p_error:.3f}")

collision_positions = free_path[collided]
plt.hist(
    collision_positions,
    bins=30,
    range=(0, A),
    density=True,
    alpha=0.7,
    edgecolor='black',
    label='Моделирование'
)
x = np.linspace(0, A, 300)
f = SIGMA * np.exp(-SIGMA * x) / (1 - np.exp(-SIGMA * A))
plt.plot(x, f, linewidth=2, label='Аналитическая плотность')
plt.xlabel("Координата первого столкновения x, см")
plt.ylabel("Условная плотность вероятности")
plt.title("Распределение первых столкновений нейтронов")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()