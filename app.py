
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Настройка страницы
# --------------------------------------------------

st.set_page_config(
    page_title="Моделирование нейтронов",
    page_icon="⚛️",
    layout="wide",
)

st.title("Моделирование взаимодействия нейтронов с мишенью")

st.write(
    "Исследование длины свободного пробега и вероятности "
    "первого столкновения методом Монте-Карло."
)


# --------------------------------------------------
# 2. Параметры симуляции
# --------------------------------------------------

st.sidebar.header("Параметры симуляции")

N = int(
    st.sidebar.number_input(
        "Количество нейтронов",
        min_value=1_000,
        max_value=1_000_000,
        value=100_000,
        step=10_000,
    )
)

SIGMA = st.sidebar.number_input(
    "Макроскопическое сечение Σ, см⁻¹",
    min_value=0.001,
    max_value=10.0,
    value=0.216,
    step=0.01,
    format="%.3f",
)

# Средняя длина свободного пробега
mean_free_path = 1 / SIGMA

st.metric(
    "Средняя длина свободного пробега λ, см",
    f"{mean_free_path:.6f}",
)

A = st.sidebar.number_input(
    "Толщина мишени a, см",
    min_value=0.001,
    max_value=100.0,
    value=0.1,
    step=0.1,
    format="%.3f",
)

seed = int(
    st.sidebar.number_input(
        "Seed генератора случайных чисел",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
    )
)

st.sidebar.caption(
    "Одинаковые параметры и seed дают воспроизводимые результаты."
)


# --------------------------------------------------
# 3. Моделирование свободного пробега
# --------------------------------------------------

rng = np.random.default_rng(seed)

# Генерация случайных чисел на интервале (0, 1).
# Защита от log(0) на случай изменения генератора.
xi = np.maximum(rng.random(N), np.finfo(float).tiny)

# Экспоненциальное распределение длины свободного пробега.
free_path = -np.log(xi) / SIGMA

# Столкновение произошло внутри мишени.
collided = free_path < A

# Первое столкновение во второй половине мишени.
second_half = (free_path >= A / 2) & (free_path < A)

# Фактический путь нейтрона внутри конечной мишени.
path_in_target = np.minimum(free_path, A)


# --------------------------------------------------
# 4. Вероятности: Монте-Карло и аналитика
# --------------------------------------------------

p_mc = collided.mean()
p_exact = -np.expm1(-SIGMA * A)

p_second_mc = second_half.mean()

# Вероятность первого столкновения во второй половине
# относительно всех падающих нейтронов.
p_second_exact = (
    np.exp(-SIGMA * A / 2)
    * (-np.expm1(-SIGMA * A / 2))
)

# Количество событий.
n_collided = int(collided.sum())
n_second_half = int(second_half.sum())


# --------------------------------------------------
# 5. Средняя длина пути
# --------------------------------------------------

mean_path_mc = path_in_target.mean()

# E[min(X, a)] = (1 - exp(-Σa)) / Σ
mean_path_exact = p_exact / SIGMA

mean_path_error = (
    path_in_target.std(ddof=1) / np.sqrt(N)
    if N > 1
    else float("nan")
)


# --------------------------------------------------
# 6. Основные результаты
# --------------------------------------------------

st.header("Результаты моделирования")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Столкновений в мишени",
        f"{n_collided:,}".replace(",", " "),
    )

    st.metric(
        "Вероятность столкновения (Монте-Карло)",
        f"{p_mc:.6f}",
    )

    st.metric(
        "Аналитическая вероятность",
        f"{p_exact:.6f}",
    )

with col2:
    st.metric(
        "Столкновений во второй половине",
        f"{n_second_half:,}".replace(",", " "),
    )

    st.metric(
        "Вероятность второй половины (Монте-Карло)",
        f"{p_second_mc:.6f}",
    )

    st.metric(
        "Аналитическая вероятность второй половины",
        f"{p_second_exact:.6f}",
    )


# --------------------------------------------------
# 7. Гистограмма распределения столкновений
# --------------------------------------------------

st.header("Распределение первых столкновений")

collision_positions = free_path[collided]

if len(collision_positions) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        collision_positions,
        bins=30,
        range=(0, A),
        density=True,
        alpha=0.7,
        edgecolor="black",
        label="Моделирование",
    )

    x = np.linspace(0, A, 300)

    # Условная плотность координаты столкновения:
    # f(x | X < a) = Σ exp(-Σx) / (1 - exp(-Σa)).
    f = SIGMA * np.exp(-SIGMA * x) / p_exact

    ax.plot(
        x,
        f,
        linewidth=2,
        label="Аналитическая плотность",
    )

    ax.set_xlabel("Координата первого столкновения x, см")
    ax.set_ylabel("Условная плотность вероятности")
    ax.set_title("Распределение первых столкновений нейтронов")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning(
        "В этой реализации нет столкновений внутри мишени. "
        "Увеличь число нейтронов или толщину мишени."
    )


# --------------------------------------------------
# 8. Оценка статистической погрешности
# --------------------------------------------------

st.header("Оценка точности моделирования")

# Стандартные ошибки оценок вероятностей.
error_p = np.sqrt(p_mc * (1 - p_mc) / N)

error_second = np.sqrt(
    p_second_mc * (1 - p_second_mc) / N
)

# Абсолютные отклонения от аналитических результатов.
delta_p = abs(p_mc - p_exact)
delta_second = abs(p_second_mc - p_second_exact)

# Отклонение в единицах стандартной ошибки.
z_p = delta_p / error_p if error_p > 0 else float("nan")

z_second = (
    delta_second / error_second
    if error_second > 0
    else float("nan")
)

# Приближённый 95%-й доверительный интервал.
ci_low = max(0.0, p_mc - 1.96 * error_p)
ci_high = min(1.0, p_mc + 1.96 * error_p)

st.subheader("Сравнение результатов")

comparison = pd.DataFrame(
    {
        "Величина": [
            "Столкновение в мишени",
            "Столкновение во второй половине",
        ],
        "Монте-Карло": [
            p_mc,
            p_second_mc,
        ],
        "Аналитически": [
            p_exact,
            p_second_exact,
        ],
        "Стандартная ошибка": [
            error_p,
            error_second,
        ],
        "Абсолютное отклонение": [
            delta_p,
            delta_second,
        ],
        "Отклонение / ошибка": [
            z_p,
            z_second,
        ],
    }
)

st.dataframe(comparison, hide_index=True, use_container_width=True)

st.subheader("95%-й доверительный интервал")

st.write(f"Нижняя граница: {ci_low:.6f}")
st.write(f"Верхняя граница: {ci_high:.6f}")
st.write(f"Аналитическая вероятность: {p_exact:.6f}")

if ci_low <= p_exact <= ci_high:
    st.success(
        "Аналитическое значение попадает в приближённый интервал."
    )
else:
    st.warning(
        "Аналитическое значение вне интервала. "
        "Такое возможно из-за случайных флуктуаций."
    )

st.caption(
    "Интервал построен нормальным приближением. "
    "При малом числе столкновений предпочтительнее интервал Уилсона."
)


# --------------------------------------------------
# 9. Средняя длина пути в мишени
# --------------------------------------------------

st.header("Средняя длина пути в мишени")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Монте-Карло, см",
        f"{mean_path_mc:.6f}",
        delta=f"{mean_path_mc - mean_path_exact:+.6f} см",
    )

with col2:
    st.metric(
        "Аналитически, см",
        f"{mean_path_exact:.6f}",
    )

st.write(
    f"Стандартная ошибка среднего: {mean_path_error:.6f} см"
)


# --------------------------------------------------
# 10. Экспорт результатов в CSV
# --------------------------------------------------

st.header("Экспорт результатов")

results = pd.DataFrame(
    {
        "Показатель": [
            "Количество нейтронов",
            "Seed генератора",
            "Макроскопическое сечение, см⁻¹",
            "Толщина мишени, см",
            "Количество столкновений",
            "Вероятность столкновения (Монте-Карло)",
            "Вероятность столкновения (аналитически)",
            "Столкновения во второй половине",
            "Вероятность второй половины (Монте-Карло)",
            "Вероятность второй половины (аналитически)",
            "Стандартная ошибка вероятности",
            "Абсолютное отклонение вероятности",
            "Средняя длина пути (Монте-Карло), см",
            "Средняя длина пути (аналитически), см",
            "Стандартная ошибка средней длины пути, см",
            "Нижняя граница 95%-го интервала",
            "Верхняя граница 95%-го интервала",
        ],
        "Значение": [
            N,
            seed,
            SIGMA,
            A,
            n_collided,
            p_mc,
            p_exact,
            n_second_half,
            p_second_mc,
            p_second_exact,
            error_p,
            delta_p,
            mean_path_mc,
            mean_path_exact,
            mean_path_error,
            ci_low,
            ci_high,
        ],
    }
)

st.dataframe(results, hide_index=True, use_container_width=True)

csv_data = results.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="Скачать результаты в CSV",
    data=csv_data,
    file_name="neutron_simulation_results.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 11. Анимация движения нейтронов
# --------------------------------------------------

st.header("Анимация движения нейтронов")

n_animation = st.slider(
    "Количество нейтронов в анимации",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

animation_speed = st.slider(
    "Скорость анимации",
    min_value=1,
    max_value=10,
    value=5,
)

start_animation = st.button("Запустить анимацию")

if start_animation:
    animation_rng = np.random.default_rng()

    animation_xi = np.maximum(
        animation_rng.random(n_animation),
        np.finfo(float).tiny,
    )

    animation_paths = -np.log(animation_xi) / SIGMA

    progress = st.progress(0)
    animation_area = st.empty()

    steps = 60

    for step in range(steps + 1):
        fraction = step / steps
        current_x = fraction * A

        fig, ax = plt.subplots(figsize=(10, 4))

        # Область мишени.
        ax.axvspan(0, A, color="gray", alpha=0.25)
        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(A, color="black", linewidth=1)

        for i, path in enumerate(animation_paths):
            if path <= current_x:
                # Нейтрон достиг точки первого столкновения.
                x_pos = path
                color = "red"
                marker = "x"
            else:
                # Нейтрон ещё движется.
                x_pos = current_x
                color = "blue"
                marker = "o"

            ax.scatter(
                x_pos,
                i,
                color=color,
                marker=marker,
                s=35,
            )

        ax.set_xlim(-0.2 * A, 1.2 * A)
        ax.set_ylim(-1, n_animation)
        ax.set_xlabel("Координата x, см")
        ax.set_ylabel("Номер нейтрона")
        ax.set_title("Движение нейтронов через мишень")
        ax.grid(alpha=0.2)

        fig.tight_layout()
        animation_area.pyplot(fig)
        plt.close(fig)

        progress.progress(step / steps)
        time.sleep(0.11 / animation_speed)

    st.success("Анимация завершена.")
