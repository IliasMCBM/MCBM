from typing import Callable, Literal
import gradio as gr
from PIL import Image
import pytesseract  # Make sure pytesseract is installed

english_examples = [
    ["How can I make a healthier version of pizza?"],
    ["How can I make my salad high in protein without adding meat?"],
    ["What’s a healthy substitute for butter when cooking?"],
    ["What are some gluten-free options for pasta?"],
    ["What can I eat with grilled chicken to make it a balanced dinner?"]
]

def clear_files():
    return "Vector Store is Not ready"

def handle_user_message(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

def extract_text_from_image(image):
    """
    Function to perform OCR on uploaded images
    and extract text to add to the vector store.
    
    Params:
      image: uploaded image
    Returns:
      extracted_text: Text extracted from the image
    """
    img = Image.open(image)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text

def make_demo(
    load_doc_fn: Callable,
    run_fn: Callable,
    stop_fn: Callable,
    update_retriever_fn: Callable,
    model_name: str,
):
    examples = english_examples
    text_example_path = "text_example_en.pdf"
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.Markdown("""<h1><center>QA over Document and Image</center></h1>""")
        gr.Markdown(f"""<center>Powered by OpenVINO and {model_name} </center>""")
        with gr.Row():
            with gr.Column(scale=1):
                docs = gr.File(
                    label="Step 1: Load text files",
                    value=[text_example_path],
                    file_count="multiple",
                    file_types=[
                        ".csv", ".doc", ".docx", ".enex", ".epub",
                        ".html", ".md", ".odt", ".pdf", ".ppt",
                        ".pptx", ".txt"
                    ],
                )
                image_upload = gr.File(
                    label="Upload an Image (Optional)",
                    file_types=[".jpg", ".jpeg", ".png"]
                )
                load_docs = gr.Button("Step 2: Build Vector Store", variant="primary")
                db_argument = gr.Accordion("Vector Store Configuration", open=False)
                with db_argument:
                    spliter = gr.Dropdown(
                        ["Character", "RecursiveCharacter", "Markdown", "Chinese"],
                        value="RecursiveCharacter",
                        label="Text Spliter",
                        info="Method used to split the documents",
                        multiselect=False,
                    )
                    chunk_size = gr.Slider(
                        label="Chunk size", value=400, minimum=50,
                        maximum=2000, step=50, interactive=True,
                        info="Size of sentence chunk",
                    )
                    chunk_overlap = gr.Slider(
                        label="Chunk overlap", value=50, minimum=0,
                        maximum=400, step=10, interactive=True,
                        info="Overlap between 2 chunks",
                    )

                langchain_status = gr.Textbox(
                    label="Vector Store Status",
                    value="Vector Store is Ready",
                    interactive=False,
                )
                do_rag = gr.Checkbox(
                    value=True, label="RAG is ON",
                    interactive=True,
                    info="Whether to do RAG for generation",
                )
                with gr.Accordion("Generation Configuration", open=False):
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature", value=0.1, minimum=0.0,
                            maximum=1.0, step=0.1, interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)", value=1.0, minimum=0.0,
                            maximum=1, step=0.01, interactive=True,
                            info="Sample from the smallest possible set of tokens whose cumulative probability exceeds top_p.",
                        )
                        top_k = gr.Slider(
                            label="Top-k", value=50, minimum=0.0,
                            maximum=200, step=1, interactive=True,
                            info="Sample from a shortlist of top-k tokens.",
                        )
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty", value=1.1, minimum=1.0,
                            maximum=2.0, step=0.1, interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=800, label="Step 3: Input Query")
                with gr.Row():
                    msg = gr.Textbox(
                        label="QA Message Box",
                        placeholder="Chat Message Box",
                        show_label=False,
                        container=False,
                    )
                    submit = gr.Button("Submit", variant="primary")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
                gr.Examples(examples, inputs=msg, label="Click on any example and press 'Submit'")
                retriever_argument = gr.Accordion("Retriever Configuration", open=True)
                with retriever_argument:
                    do_rerank = gr.Checkbox(value=True, label="Rerank searching result", interactive=True)
                    hide_context = gr.Checkbox(value=True, label="Hide searching result in prompt", interactive=True)
                    search_method = gr.Dropdown(
                        ["similarity_score_threshold", "similarity", "mmr"],
                        value="similarity_score_threshold",
                        label="Searching Method",
                        info="Method used to search vector store",
                        multiselect=False,
                        interactive=True,
                    )
                    score_threshold = gr.Slider(
                        0.01, 0.99, value=0.5, step=0.01,
                        label="Similarity Threshold",
                        info="Only working for 'similarity score threshold' method",
                        interactive=True,
                    )
                    vector_rerank_top_n = gr.Slider(
                        1, 10, value=2, step=1, label="Rerank top n",
                        info="Number of rerank results", interactive=True,
                    )
                    vector_search_top_k = gr.Slider(
                        1, 50, value=10, step=1, label="Search top k",
                        info="Search top k must >= Rerank top n", interactive=True,
                    )

        docs.clear(clear_files, outputs=[langchain_status], queue=False)
        load_docs.click(
            fn=load_doc_fn,
            inputs=[docs, spliter, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
            queue=False,
        )
        submit_event = msg.submit(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            run_fn,
            [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
            chatbot,
            queue=True,
        )
        submit_click_event = submit.click(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            run_fn,
            [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
            chatbot,
            queue=True,
        )
        image_upload.upload(
            fn=lambda image: extract_text_from_image(image),
            inputs=image_upload,
            outputs=langchain_status,
        )
        stop.click(
            fn=stop_fn, inputs=None, outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)
        
    return demo
