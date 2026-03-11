module load anaconda3/2023.09
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.37.200 3138
source /APP/u22/ai_x86/toolshs/network.sh hitsz_xdeng_2 Db7AQXLzNov04S7i
conda create -n game python=3.10 -y
conda init bash
source ~/.bashrc
conda activate game

pip install --upgrade pip setuptools wheel --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip uninstall -y torch torchvision torch-tensorrt \
    flash_attn transformer-engine \
    cudf dask-cuda cugraph cugraph-service-server cuml raft-dask cugraph-dgl cugraph-pyg dask-cudf
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


pip install "numpy==1.26.4" "optree>=0.13.0" "spacy==3.7.5" "weasel==0.4.1" \
    "transformer-engine[pytorch]==2.2.0" "megatron-core==0.11.0" "deepspeed==0.16.4" \
    --no-build-isolation # 不加入--no-build-isolation会报错
    # 使用阿里云会报错--trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install vllm==0.8.4 \
    --trusted-host mirrors.aliyun.com --index-url https://mirrors.aliyun.com/pypi/simple/
#警告：新安装的包强制升级了 numpy 到 2.x 版本，但这导致你环境里原有的 thinc 包（通常是 spaCy 的依赖）可能无法工作，因为它只支持旧版的 numpy
#ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#thinc 8.2.5 requires numpy<2.0.0,>=1.19.0; python_version >= "3.9", but you have numpy 2.2.6 which is incompatible.
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 32" \
    "git+https://github.com/NVIDIA/apex.git@25.04"


#=================================================================
# pip install -r requirements_common.txt 

#==========================================
# 一句解决
# pip install -r requirements_torch260_vllm.txt
# 但是安装flash-attn遇到了问题
# 手动解决：pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install transformer-engine[pytorch]==2.2.0 deepspeed==0.16.4 vllm==0.8.4
# pip install -r requirements_common.txt 

# 测试： sh run_frozen_lake.sh