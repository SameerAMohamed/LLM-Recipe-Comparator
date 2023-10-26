from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# To run on GPU: https://python.langchain.com/docs/integrations/llms/llamacpp#installation-with-openblas--cublas--clblast

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def local_llm(path, text):# Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=path, # Downloaded from https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/tree/main on commit 4458acc949de0a9914c3eab623904d4fe999050a
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=1024*8,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
    )

    a = llm(text)
    return a

local_llm("/models/llama-2-13b-chat.Q4_0.gguf", "The following is a list of ingredients and instructions to make a recipe of chocolate chip cookies.")