# pdf-chatbot

Writing up a chatbot that can RAG from multiple pdfs

Doing some follow-along coding with [this video](https://www.youtube.com/watch?v=dXxQ0LR-3Hg&t=75s)

## install

```bash
poetry init
poetry shell
```

Then you will need a `.env` file that has

```txt
OPENAI_API_KEY=<key>
HUGGINGFACEHUB_API_TOKEN=<key>
```

It is easiest to do it this way, since LangChain will look for these enviornment variables specifically, and the app checks for them and surfaces them as env variables if they are in the `.env` file.

## running

```bash
streamlit run app.py
```

## Models

Embedding model is either

- OpenAI ada2
- [instructor-xl](https://huggingface.co/hkunlp/instructor-xl)

LLM

- Open AI gpt-3.5 (or whatever you have access to)

## Dev work

To get reloading, you can add

```toml
[server]
runOnSave = true
```

to the the .streamlit/config.toml to get it to reload on save. However, this causes it to periodically re-submit the user prompt, so not great for trying to chat.
