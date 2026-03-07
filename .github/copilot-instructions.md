# Copilot Instructions

- **Tech stack:** Python, Streamlit, pandas, numpy, matplotlib, scikit-learn, imbalanced-learn.
- Keep code modular and importable. Prefer small pure functions over large scripts.
- Preserve parity first: match existing formulas, plots, defaults, and naming before refactoring behavior.
- Put metric definitions in dedicated registry/modules. New metrics should be addable by editing one metric file only.
- Do not hardcode UI logic inside metric implementations.
- Use typed functions, dataclasses where useful, and explicit return values.
- Avoid hidden state. Pass config objects/params explicitly.
- Keep plotting code separated from computation code.
- Handle undefined metric values with `np.nan`; never silently coerce.
- For expensive computations, add caching and progress reporting suitable for Streamlit.
- Prefer deterministic behavior: fixed random seeds, stable ordering, reproducible outputs.
- Validate user inputs and fail with clear exceptions/messages.
- Do not introduce new dependencies unless necessary.
- Keep comments minimal and high-signal. No decorative comments.
- When changing formulas or defaults, preserve backward compatibility unless explicitly asked.
- Favor readability for future extension over micro-optimizations.
- Add new metrics using the existing registry pattern and ensure they automatically appear in dropdowns/plots when possible.