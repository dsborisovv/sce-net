# Single-Embedding SCE: как обучить общую совместимость и считать cosine на инференсе

Ниже — практичный вариант под ваш запрос:

- `train_triplets` уже содержит хорошие `(anchor, positive, negative)`.
- В трейне мы учим **general compatibility** через triplet objective.
- На инференсе у каждого айтема есть **один** эмбеддинг `z_i`.
- Совместимость пары считается как `cosine(z_i, z_j)` без pairwise forward сложной модели.

Реализация: `single_embedding_sce.py`.

## 1) Архитектура
`SingleEmbeddingSCENet` делает следующее:

1. Изображение → базовый CLIP эмбеддинг `z_base`.
2. `router(z_base)` предсказывает веса условий `alpha` (softmax по `M` conditions).
3. `condition_prototypes` (матрица `M x D`) агрегируются через `alpha` в `cond_vec`.
4. Итоговый item-вектор:

\[
z = \mathrm{norm}(z_{base} + s \cdot cond\_vec)
\]

где `s = cond_scale`.

Ключевой момент: роутинг считается по **одному айтему**, а не по паре. Поэтому на инференсе не нужен pairwise attention-проход.

## 2) Лосс
`compatibility_loss(...)` объединяет:

- **Triplet cosine margin**: `cos(a,p)` должен быть больше `cos(a,n)` на margin.
- **Auxiliary pair BCE**: пары `(a,p)=1`, `(a,n)=0` на логитах от cosine.
- **Orthogonality regularization** для conditions, чтобы `M` условий не схлопывались в одно.
- **Entropy regularization** для `alpha`, чтобы роутер использовал условия более стабильно.

Идея: модель учит паттерны совместимости при тренировке, но в проде даёт компактный item embedding.

## 3) Как интегрировать в текущий pipeline
Минимально:

1. Используйте ваш существующий triplet dataloader.
2. На батче получите `(z_a, alpha_a)`, `(z_p, alpha_p)`, `(z_n, alpha_n)`.
3. Посчитайте `loss, stats = compatibility_loss(...)`.
4. Стандартно `backward/step`.

После обучения:

1. Один раз прогоните весь каталог через `encode_items(...)`.
2. Сохраните `item_id -> embedding`.
3. В проде используйте `cosine_compatibility(...)`.

## 4) Почему это решает вашу задачу
Вы хотели «что-то как SCENet с conditions, но без pair scoring на инференсе».

Этот вариант делает именно это:
- conditions есть и учатся в трейне;
- но их влияние уже «вплавляется» в `z_i` каждого айтема;
- inference = дешёвый cosine по кэшированным векторам.

## 5) Что валидировать в экспериментах
Помимо AUC/PR-AUC, важно смотреть:
- Recall@K по кросс-категориям (top→bottom, dress→shoes);
- margin `mean(cos_ap - cos_an)` на вале;
- долю используемых conditions (распределение `alpha`).

Если `alpha` всегда пиковый в один и тот же condition — увеличить `entropy_weight`.
Если условия стали одинаковыми — увеличить `orthogonality_weight`.
