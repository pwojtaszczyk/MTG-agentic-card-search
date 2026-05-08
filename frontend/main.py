from __future__ import annotations

from fastapi import FastAPI
import gradio as gr

from frontend.app import build_demo


def create_app() -> FastAPI:
    """Create an ASGI app that mounts the Gradio UI."""
    api = FastAPI(title="MTG Agentic Search Frontend")
    demo = build_demo()
    return gr.mount_gradio_app(api, demo, path="/")


app = create_app()

