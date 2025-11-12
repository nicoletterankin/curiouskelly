## Presets Index

Identity
- `identity_front.yaml` — 8K square, studio neutral, front view
- `identity_three_quarter.yaml` — 8K square, studio neutral, 3/4 view
- `identity_profile.yaml` — 8K square, studio neutral, profile view

Expressions (front, studio neutral, 8K square)
- `expressions_neutral.yaml`
- `expressions_happy.yaml`
- `expressions_surprised.yaml`
- `expressions_sad.yaml`
- `expressions_angry.yaml`
- `expressions_thinking.yaml`

Hair/Plates
- `hair_alpha_plate.yaml` — 8K 16:9 plate with back/rim light for edge detail

Usage
```powershell
powershell -ExecutionPolicy Bypass -File ..\scripts\generate_kelly_asset.ps1 -Preset identity_front.yaml
# or batch everything
powershell -ExecutionPolicy Bypass -File ..\scripts\generate_kelly_batch.ps1
```


