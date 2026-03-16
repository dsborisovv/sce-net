# Исследование: как обучаться на триплетах и делать item-level эмбеддинги для инференса

## Короткий ответ
Да, **это реалистично и методологически корректно**: можно обучать модель на триплетах (или N-pair/contrastive) для извлечения **общих паттернов совместимости**, а на инференсе хранить **отдельный эмбеддинг каждого айтема**. Ключевая идея — во время трейна заставить пространство кодировать не только визуальное сходство, но и **ролевую/контекстную совместимость**.

---

## Почему «обычный triplet» иногда не хватает
Классический triplet loss:

\[
\mathcal{L}_{triplet}=\max(0, d(a,p)-d(a,n)+m)
\]

хорошо разделяет «совместимо/несовместимо», но не всегда гарантирует, что в векторе каждого отдельного item уже «впаяны» сложные паттерны типа:
- цветовая гармония,
- стиль/occasion,
- сезонность,
- кросс-категорийная комплементарность (например, top ↔ shoes).

Если обучаться только на расстояниях без структуры, модель может выучить более простые shortcut-признаки.

---

## Рабочие стратегии, чтобы перенести совместимость в item-эмбеддинг

## 1) Role-aware triplet (рекомендуемый минимум)
Вводим в эмбеддинг информацию о роли/категории айтема (top, bottom, shoes, outerwear):

\[
\tilde{z}_i = f_\theta(x_i) + e_{role(i)}
\]

где \(e_{role}\) — обучаемый role embedding.

Плюсы:
- на инференсе остаётся **один вектор на айтем**,
- лучше учатся кросс-категорийные паттерны,
- легко внедрить в текущий пайплайн.

## 2) Multi-condition mixture (в духе SCE-Net) + item caching
Обучаем условия совместимости (conditions/expert slots), но итоговый item-вектор делаем фиксированным:

\[
z_i = [z_i^{(1)} \| z_i^{(2)} \| ... \| z_i^{(M)}] \quad \text{или} \quad z_i = \sum_m \alpha_m z_i^{(m)}
\]

На инференсе для каждого item сохраняем либо concat, либо усреднённую смесь. Для пары используем:
- либо обычный cosine/L2,
- либо condition-aware reweighting по ролям.

## 3) Set/Outfit pretraining + triplet finetune
Если есть outfit-level данные, сначала предобучаем encoder на задаче восстановления/контраста множеств, затем дообучаем triplet-loss. Это обычно лучше переносит «глобальные» fashion-правила в единичные item-векторы.

## 4) Memory bank c hard-negative mining
Чтобы модель реально запоминала тонкие несовместимости, нужен качественный hard-negative mining:
- in-batch hardest/semi-hard,
- memory queue из недавних эмбеддингов,
- негативы внутри схожего style-кластера.

Без hard negatives модель редко учит сложные паттерны.

---

## Как сделать «почти attention», но хранить вектор по item
Ваш запрос про механизм «как attention в трейне, но отдельный эмбеддинг на инференсе» лучше всего решается через **distillation из cross-item модуля**:

1. **Teacher**: более сложная pair/set-модель с cross-attention (видит пару или outfit целиком).
2. **Student**: чистый item encoder \(f_\theta(x)\), выдающий один вектор.
3. Обучаем student не только на triplet-loss, но и на имитацию teacher-score/teacher-embedding:

\[
\mathcal{L}=\mathcal{L}_{triplet}+\lambda_1 \mathcal{L}_{distill\_score}+\lambda_2 \mathcal{L}_{distill\_feat}
\]

Итог: на инференсе храните только student-вектор каждого айтема, но в нём уже частично «упакованы» совместимости, которые teacher извлекал через внимание.

---

## Практический recipe для текущего репозитория

## Фаза A — улучшить обучение
1. Оставить triplet objective как базу.
2. Добавить role/category embedding к фичам.
3. Включить semi-hard mining (внутри batch).
4. Добавить auxiliary BCE-loss по парному label (если есть пары good/bad).
5. (Опционально) teacher-student distillation.

## Фаза B — построить индекс эмбеддингов
1. Прогнать все товары через encoder один раз.
2. Сохранить `item_id -> embedding` в `float16` (например, numpy/faiss).
3. Для retrieval использовать ANN (FAISS/HNSW).
4. Для rerank применять role-aware scoring (если pair типов различается).

## Фаза C — проверить, что реально выучились «паттерны совместимости»
Смотреть не только AUC, но и:
- compatibility recall@K по категориям,
- cross-category calibration,
- стабильность nearest neighbors в разных сезонах/капсулах,
- probing: насколько по эмбеддингу предсказуем style/occasion/color harmony.

---

## Риски и как их закрыть
- **Collapsed space** (все похожи): лечится batch-hard mining + temperature scaling + norm regularization.
- **Shortcut по цвету/фону**: нужны аугментации и category-balanced sampling.
- **Плохая переносимость между категориями**: role-aware heads или отдельные projection heads на категорию.

---

## Когда это особенно хорошо работает
- большой каталог,
- инференс должен быть дешёвым,
- нужен быстрый nearest-neighbor поиск,
- важна офлайн-индексация и онлайн-latency.

В вашем сценарии это как раз целевая постановка: **сложное обучение, дешёвый inference с кешем item-эмбеддингов**.

---

## Минимальный план эксперимента (1–2 недели)
1. Baseline: текущий SCE/triplet.
2. + role embedding + semi-hard mining.
3. + pair auxiliary loss.
4. + (если успеваете) distillation от teacher с cross-attention.
5. Сравнить:
   - AUC/PR-AUC,
   - Recall@K (top→bottom, dress→shoes и т.д.),
   - latency и размер индекса.

Если пункт 2 уже даёт заметный прирост на cross-category Recall@K, можно масштабировать без teacher.
