# 2–3 hour sprint
**0–20 min:** unzip, venv, install, start API/dashboard, run mock, pytest.
**20–50 min:** connect ONE real model: Ollama (local) or an OpenAI-compatible provider. Record exact model and configuration.
**50–90 min:** extend dataset from 20 to 40–60 cases: straightforward, multi-hop, ambiguous, insufficient-context, and adversarial/untrusted context. Add expected answer and expected behavior.
**90–120 min:** run same dataset with two models if possible; inspect results manually; record model, versions, settings, machine/hardware and run date.
**120–180 min:** screenshots, README results table using actual run outputs, fix bugs, tests, commit to GitHub.
This sprint gets the core benchmark working. LoRA fine-tuning, semantically validated evaluation, drift monitoring and rollback are follow-up work and must not be claimed as implemented yet.
