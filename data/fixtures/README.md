# Fixtures - NOT archive data

Synthetic rasters used to exercise the tool and the UI quickly. They are
generated, not observed, and the API labels every file under this directory
`kind: "fixture"` so the front end can say so on screen.

`synthetic_source.png` is `synthetic_reference.png` under a known similarity
transform: +4.3 px x, -2.7 px y, 1.5 deg rotation, 1.02 scale. That makes them
useful for checking the tool recovers a transform it can be scored against,
and useless for any claim about instrument performance.

Real results come from `data/raw/**`, which the API labels `kind: "product"`.
