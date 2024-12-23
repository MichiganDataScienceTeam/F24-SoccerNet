from setuptools import setup, find_packages

setup(
    name='SoccerAnalysis',             
    version='0.1.0',                
    description='A soccer analysis package using Roboflow and tracking systems to accurately track a soccer game',
    author='Antonio Capdevielle, Shiva Chandran',
    author_email='acapdevi@umich.edu',
    packages=find_packages(),         
    install_requires=[                
        'absl-py==2.1.0',
        'aiohttp==3.9.5',
        'aioresponses==0.7.6',
        'aiosignal==1.3.1',
        'annotated-types==0.7.0',
        'anyio==4.4.0',
        'APScheduler==3.10.1',
        'astunparse==1.6.3',
        'attrs==23.2.0',
        'awscli==1.33.12',
        'backoff==2.2.1',
        'boto3==1.34.123',
        'botocore==1.34.130',
        'cachetools==5.3.3',
        'certifi==2024.6.2',
        'cffi==1.16.0',
        'chardet==4.0.0',
        'charset-normalizer==3.3.2',
        'click==8.1.7',
        'cloudpickle==1.3.0',
        'colorama==0.4.4',
        'coloredlogs==15.0.1',
        'contourpy==1.2.1',
        'cryptography==42.0.8',
        'cycler==0.12.1',
        'Cython==3.0.0',
        'dataclasses-json==0.6.7',
        'decorator==5.1.1',
        'defusedxml==0.7.1',
        'distro==1.9.0',
        'dm-sonnet==2.0.0',
        'dm-tree==0.1.8',
        'docker==6.1.3',
        'docutils==0.16',
        'efficientnet==1.0.0',
        'fastapi==0.110.3',
        'filelock==3.15.3',
        'flatbuffers==24.3.25',
        'fonttools==4.53.0',
        'frozenlist==1.4.1',
        'gast==0.6.0',
        'google-api-core==2.19.0',
        'google-api-python-client==2.134.0',
        'google-auth==2.30.0',
        'google-auth-httplib2==0.2.0',
        'google-cloud-core==2.4.1',
        'google-cloud-storage==2.17.0',
        'google-crc32c==1.5.0',
        'google-pasta==0.2.0',
        'google-resumable-media==2.7.1',
        'googleapis-common-protos==1.63.1',
        'GPUtil==1.4.0',
        'grpcio==1.65.4',
        'h11==0.14.0',
        'h5py==3.11.0',
        'httpcore==1.0.5',
        'httplib2==0.22.0',
        'httpx==0.27.0',
        'humanfriendly==10.0',
        'idna==3.7',
        'image-classifiers==1.0.0',
        'imageio==2.34.1',
        'inference==0.12.1',
        'inference-sdk==0.12.1',
        'iniconfig==2.0.0',
        'Jinja2==3.1.4',
        'jmespath==1.0.1',
        'joblib==1.4.2',
        'jsonschema==4.22.0',
        'jsonschema-specifications==2023.12.1',
        'keras==3.4.1',
        'Keras-Applications==1.0.8',
        'kiwisolver==1.4.5',
        'lazy_loader==0.4',
        'libclang==18.1.1',
        'Markdown==3.6',
        'markdown-it-py==3.0.0',
        'MarkupSafe==2.1.5',
        'marshmallow==3.21.3',
        'matplotlib==3.9.0',
        'mdurl==0.1.2',
        'ml-dtypes==0.4.0',
        'mpmath==1.3.0',
        'multidict==6.0.5',
        'mypy-extensions==1.0.0',
        'namex==0.0.8',
        'networkx==3.3',
        'numpy==1.26.4',
        'onnxruntime==1.15.1',
        'openai==1.35.2',
        'opencv-contrib-python==4.10.0.84',
        'opencv-python==4.8.0.76',
        'opencv-python-headless==4.10.0.84',
        'opt-einsum==3.3.0',
        'optree==0.12.1',
        'packaging==24.1',
        'pandas==2.2.2',
        'pendulum==3.0.0',
        'piexif==1.1.3',
        'pillow==10.3.0',
        'pluggy==1.5.0',
        'prettytable==3.10.0',
        'prometheus-fastapi-instrumentator==6.0.0',
        'prometheus_client==0.20.0',
        'proto-plus==1.24.0',
        'protobuf==4.25.3',
        'psutil==6.0.0',
        'PuLP==2.8.0',
        'py-cpuinfo==9.0.0',
        'pyasn1==0.6.0',
        'pyasn1_modules==0.4.0',
        'pybase64==1.3.2',
        'pycparser==2.22',
        'pydantic==2.7.4',
        'pydantic_core==2.18.4',
        'pydot==2.0.0',
        'Pygments==2.18.0',
        'pyparsing==3.1.2',
        'pytest==8.2.2',
        'pytest-asyncio==0.21.1',
        'python-dateutil==2.9.0.post0',
        'python-dotenv==1.0.1',
        'python-magic==0.4.27',
        'pytz==2024.1',
        'PyYAML==6.0.1',
        'redis==5.0.6',
        'referencing==0.35.1',
        'requests==2.31.0',
        'requests-toolbelt==1.0.0',
        'rich==13.5.2',
        'roboflow==1.1.33',
        'rpds-py==0.18.1',
        'rsa==4.7.2',
        's3transfer==0.10.1',
        'scikit-image==0.24.0',
        'scikit-learn==1.5.1',
        'scipy==1.13.1',
        'segmentation-models==1.0.1',
        'shapely==2.0.1',
        'six==1.16.0',
        'skypilot==0.5.0',
        'sniffio==1.3.1',
        'starlette==0.37.2',
        'structlog==24.2.0',
        'supervision==0.21.0',
        'sympy==1.12.1',
        'tabulate==0.9.0',
        'tensorboard==2.17.0',
        'tensorboard-data-server==0.7.2',
        'tensorflow==2.17.0',
        'tensorflow-io-gcs-filesystem==0.37.1',
        'tensorflow-probability==0.11.0',
        'termcolor==2.4.0',
        'threadpoolctl==3.5.0',
        'tifffile==2024.6.18',
        'time-machine==2.14.1',
        'tqdm==4.66.4',
        'typer==0.9.0',
        'typing-inspect==0.9.0',
        'typing_extensions==4.12.2',
        'tzdata==2024.1',
        'tzlocal==5.2',
        'uritemplate==4.1.1',
        'urllib3==1.26.19',
        'wcwidth==0.2.13',
        'websocket-client==1.8.0',
        'Werkzeug==3.0.3',
        'wrapt==1.16.0',
        'yarl==1.9.4',
        'zxing-cpp==2.2.0',
    ],
    classifiers=[                     
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='==3.11.4',        
)
