"""Local VLM labeler (Qwen3-VL via MLX) used only to bootstrap training labels."""
from functools import cached_property

DEFAULT_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


class VLM:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id

    @cached_property
    def _loaded(self):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model, processor = load(self.model_id)
        return model, processor, load_config(self.model_id)

    def ask(self, image_path: str, question: str, max_tokens: int = 60) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor, config = self._loaded
        prompt = apply_chat_template(processor, config, question, num_images=1)
        result = generate(model, processor, prompt, [image_path], max_tokens=max_tokens,
                          temperature=0.0, verbose=False)
        return (result.text if hasattr(result, "text") else str(result)).strip()

    def ask_yes_no(self, image_path: str, question: str) -> bool | None:
        text = self.ask(image_path, question, max_tokens=5).lower()
        if text.startswith(("yes", "true")):
            return True
        if text.startswith(("no", "false")):
            return False
        return None



SCENE_PROMPT = (
    "This is a frame from a tennis broadcast. Is it the MAIN GAME CAMERA: the standard elevated "
    "wide shot from behind one baseline that shows the entire court (both baselines and both "
    "sidelines visible), used during rallies? Close-ups of players, crowd, benches, side angles, "
    "net-level cameras, aerial shots, full-screen graphics and Hawk-Eye animations are NOT the "
    "main game camera. Answer with one word: yes or no."
)
