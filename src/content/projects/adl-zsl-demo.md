---
title: "Zero-Shot ADL Recognition Demo"
type: "project"
status: "Live (paper forthcoming)"
order: 6
links:
  demo: "https://huggingface.co/spaces/William-heyden/adl-zsl-demo"
summary: "A live demo of zero-shot recognition of activities of daily living from wrist-worn IMU motion alone — the model is trained on a subset of activities and never sees a single example of the rest, only a text description of what they are."
---

Wrist motion — accelerometer, gyroscope, magnetometer — is enough to recognize activities of daily living (ADLs) like brushing teeth, putting on a jacket, or washing dishes, without a labeled training example of every activity you'd want to recognize. This demo lets you test that directly: click through, and the model classifies a real recorded wrist-motion segment, including for activities it was never trained on — it only ever received a text description of what they are.

<iframe
  src="https://william-heyden-adl-zsl-demo.hf.space"
  title="Zero-shot ADL recognition demo"
  width="100%"
  height="620"
  style="border: 1px solid var(--border); border-radius: var(--radius);"
  loading="lazy"
></iframe>

On a small screen, [open the demo directly](https://huggingface.co/spaces/William-heyden/adl-zsl-demo) instead.

The paper behind this is still in progress — this page will link to it once it's out.
