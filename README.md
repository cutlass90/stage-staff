To reproduce the results you need GPU with at least 16GB VRAM

#### Environment preparation
Create virtual env:
```bash
python3 -m venv venv
. venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Export your openAI API key
```
export OPENAI_API_KEY="<your key here>"
```

#### Stage generation
In file stage_generation.py you will find StageGenerator - a class-wrapper that incorporate all mechanics needed to generate stages.
<br>In file stage_generation_demo.py is UI interface to play with stage generation.
<br>To run UI demo:
```
streamlit run stage_generation_demo.py
```

#### Stage description
In file stage_description.py you will find ImageDescriptor - a class-wrapper that incorporate all mechanics needed to describe stages.
<br>In file stage_description_streamlit_demo.py is UI interface to play with stage description.
<br>To run UI demo:
```
streamlit run stage_description_streamlit_demo.py
```