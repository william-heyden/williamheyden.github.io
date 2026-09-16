---
title: "CAST Interactive Demo"
type: "project"
status: "Live"
order: 5
basedOn: ["cast"]
links:
  demo: "https://huggingface.co/spaces/William-heyden/cast-demo"
summary: "A live, in-browser demo of CAST's closed-form weight injection — describe a class a pretrained ImageNet classifier has never seen, upload a photo, and watch it get recognized with zero training."
---

Pick a pretrained ImageNet-1k classifier (ResNet18, MobileNetV2, or EfficientNet-B0), describe a class it's confirmed to have never seen — without using its name — check the semantic extrapolation residual ρ_u, then upload a photo and hit "Inject weight & classify." The whole thing runs CAST's closed-form ridge bridge live, no training or gradient steps involved.

<iframe
  src="https://william-heyden-cast-demo.hf.space"
  title="CAST zero-shot classifier extension demo"
  width="100%"
  height="1450"
  style="border: 1px solid var(--border); border-radius: var(--radius);"
  loading="lazy"
></iframe>

On a small screen, the controls above are easier to use in their own tab: [open the demo directly](https://huggingface.co/spaces/William-heyden/cast-demo).
