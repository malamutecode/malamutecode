Readme regarding model loading.

1) Enable developer mode for caching model:
   https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

2) GPTQ quantized model need additional libraries.
   a) optimum==1.22.0
   b) auto-gptq
        Install from pip or with manual build from 
    Download AutoGPTQ from github: git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
    Build from setup.py:
    python setup.py build
    python setup.py install