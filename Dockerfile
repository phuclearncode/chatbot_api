FROM python:3.12.7-slim

WORKDIR /chatbot_api

RUN pip install torch 

RUN pip install transformers

RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoModelForCausalLM.from_pretrained('huyhoangt2201/llama-3.2-1b-sql_finetuned_multitableJidouka2_1.0_977_records_mix_fix_210_records_merged'); \
    AutoTokenizer.from_pretrained('huyhoangt2201/llama-3.2-1b-sql_finetuned_multitableJidouka2_1.0_977_records_mix_fix_210_records_merged')";

COPY . /chatbot_api

RUN  pip install --no-cache-dir -r /chatbot_api/requirements.txt 

EXPOSE 7860

CMD ["python", "app.py"]

