
# Contributing Guidelines

## Branch Naming
- `feature/<name>` — для новых функций или моделей
- `exp/<model>-<shortdesc>` — для экспериментов с гиперпараметрами
- `fix/<name>` — для исправлений багов
- `docs/<name>` — для обновления документации

**Примеры:**
```text
feature/gnn-attention
exp/model-a-baseline-lr001
fix/dataloader-sampler
docs/readme-update
```

## Commit Messages
Формат: `<type>: <short description>`

**Types:**
- `feat:` — новая функция (модель, пайплайн)
- `fix:` — исправление ошибки
- `exp:` — результаты или конфиг эксперимента
- `docs:` — документация
- `refactor:` — рефакторинг кода
- `test:` — добавление тестов
- `chore:` — прочие изменения

**Примеры:**
```text
feat: add model B with attention mechanism
fix: correct grad_scaler in train.py
exp: model A with label smoothing 0.1
docs: update SETUP instructions
```

## Pull Request Guidelines
Обязательно включите в описание:
1. **Что сделано**
2. **Как запускать**
3. **Гистограмма результатов / Логи** (если применимо)

## 🔒 LOCKED Files
Следующие файлы образуют ядро проекта. Их изменение требует **обязательного review**:
- `src/preprocess.py`
- `src/data.py`
- `src/train.py`
- `src/eval.py`
- `splits/folds.json`
- `requirements.txt`

Изменения в этих файлах могут сломать совместимость или сломать воспроизводимость для других разработчиков!

