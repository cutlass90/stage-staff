import streamlit as st
from stage_generation import StageGenerator
from PIL import Image

st.set_page_config(layout="wide")
st.title("Stage Generator Demo")

PROMPT = "a city street filled with lots of tall buildings, muted deep neon color, metal slug concept art, chasm, 8 k highly detailed ❤🔥 🔥 💀 🤖 🚀, lofi artstyle, background ( dark _ smokiness ), hand - drawn 2 d art, industrial space, 21:9, garage, by senior artist, unknown space, dark blurry background"
DEVICE = "cuda:1"


def main():
    if 'stage_generator' not in st.session_state:
        with st.spinner(text="Loading models. Please wait..."):
            st.session_state['stage_generator'] = StageGenerator(device=DEVICE)
    prompt = st.text_input("Prompt", PROMPT)
    controlnet_conditioning_scale = st.number_input('controlnet_conditioning_scale', min_value=0.0, max_value=1.0,
                                                    value=0.5, step=0.01)
    steps = st.number_input('steps', min_value=1, max_value=100, value=30, step=1)

    input_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if input_image is not None:
        col1, col2, col3 = st.columns(3)
        image = Image.open(input_image)
        with col1:
            st.image(image, use_column_width=True)
        if st.button("RUN"):
            with st.spinner(text="Working. Please wait..."):
                depth_image = st.session_state['stage_generator'].get_depth_map(image)
                with col2:
                    st.image(depth_image, use_column_width=True)
                result = st.session_state['stage_generator'](prompt, depth_image, steps, controlnet_conditioning_scale)
                with col3:
                    st.image(result, use_column_width=True)


if __name__ == "__main__":
    main()
