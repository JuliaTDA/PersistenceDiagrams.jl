# MLJ Models

Use `recommended_ranges(model)` for suggested search specifications and
`tuning_ranges(model)` after loading MLJ/MLJBase to build native ranges, including
nested TDA models in a pipeline. Put the full PH/vectorizer/classifier pipeline
inside `TunedModel` and use outer CV to estimate generalization. Native descriptors
are also available through `PersistenceDescriptorVectorizer`; its learned grids
and ATOL codebooks are fitted within each training fold.

```@docs
TDAPersistenceDiagrams.PersistenceImageVectorizer
```

```@docs
TDAPersistenceDiagrams.PersistenceCurveVectorizer
```

```@docs
TDAPersistenceDiagrams.PersistenceLandscapeVectorizer
```
