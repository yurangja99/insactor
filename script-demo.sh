export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=1

streamlit run tools/demo.py --server.port 8501
