# VisionABR

VisionABR is a gateway that sits in front of a vLLM server running a vision-language model. When the server is quiet, every image goes through at full resolution. When load rises, it lowers image resolution per request, but only for image types where measurements show it is safe, while protecting text-heavy images and documents.

Like adaptive bitrate in video streaming, but for vision language model serving. Degrade gracefully under load instead of falling over.

## Status

Early development, v0.0.1.

This version ships only size-budget helpers, gateway in development.

## Install

```bash
pip install visionabr
```

## Example

```python
from visionabr import max_pixels_for_size

max_pixels_for_size("~512")  # 401408, the max_pixels cap for a 512 image-token budget
```

## License

MIT. See [LICENSE](LICENSE).