Este repositorio implementa estrategias de trading algorítmico basadas en Deep Learning, utilizando dos arquitecturas principales: MLP y CNN. Todo está diseñado para ejecutarse automáticamente desde el archivo main.py, dentro de la carpeta Proyecto-3.

El flujo completo —descarga de datos, ingeniería de características, estandarización, entrenamiento, evaluación y drift analysis— se ejecuta con solo correr el main.py.

 Flujo de trabajo

El Backtesting está comentado por defecto, debido a que:

Se requiere primero seleccionar los mejores modelos entrenados desde la interfaz de MLflow Tracking UI.

Una vez que el usuario haya revisado los experimentos en MLflow y elegido los modelos que desea usar:

Ingresar a la carpeta del modelo en MLflow:

mlruns/<experiment_id>/models/<model_name>/version-1/


Abrir el archivo meta.yaml

Copiar el atributo:

source: <ruta_al_modelo>


Pegar la ruta en main.py en la sección correspondiente:

ruta_mlp si el modelo elegido fue el MLP

ruta_cnn si fue el modelo CNN

Descomentar las líneas del Backtesting dentro de main.py

 Una vez hecho esto, el script completo se ejecutará correctamente.

 las librerias usadas para este projecto fueron las siguientes:
 
 absl-py==2.3.1
alembic==1.17.0
annotated-doc==0.0.3
annotated-types==0.7.0
anyio==4.11.0
astunparse==1.6.3
attrs==25.4.0
Authlib==1.6.5
backports.tarfile==1.2.0
beartype==0.22.4
beautifulsoup4==4.14.2
blinker==1.9.0
cachetools==6.2.1
certifi==2025.10.5
cffi==2.0.0
charset-normalizer==3.4.4
click==8.3.0
cloudpickle==3.1.1
contourpy==1.3.2
cryptography==45.0.7
curl_cffi==0.13.0
cycler==0.12.1
cyclopts==4.0.0
databricks-sdk==0.70.0
diskcache==5.6.3
dnspython==2.8.0
docker==7.1.0
docstring_parser==0.17.0
docutils==0.22.2
email-validator==2.3.0
exceptiongroup==1.3.0
fastapi==0.120.0
fastmcp==2.13.0.1
Flask==3.1.2
flask-cors==6.0.1
flatbuffers==25.9.23
fonttools==4.60.1
frozendict==2.4.6
gast==0.6.0
gitdb==4.0.12
GitPython==3.1.45
google-auth==2.41.1
google-pasta==0.2.0
graphene==3.4.3
graphql-core==3.2.6
graphql-relay==3.2.0
greenlet==3.2.4
grpcio==1.76.0
h11==0.16.0
h5py==3.15.1
httpcore==1.0.9
httpx==0.28.1
httpx-sse==0.4.3
idna==3.11
importlib_metadata==8.7.0
isodate==0.7.2
itsdangerous==2.2.0
jaraco.classes==3.4.0
jaraco.context==6.0.1
jaraco.functools==4.3.0
Jinja2==3.1.6
joblib==1.5.2
jsonschema==4.25.1
jsonschema-path==0.3.4
jsonschema-specifications==2025.9.1
keras==3.11.3
keyring==25.6.0
kiwisolver==1.4.9
lazy-object-proxy==1.12.0
libclang==18.1.1
Mako==1.3.10
Markdown==3.9
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.7
mcp==1.19.0
mdurl==0.1.2
ml_dtypes==0.5.3
mlflow==3.4.0
mlflow-skinny==3.4.0
mlflow-tracing==3.4.0
more-itertools==10.8.0
multitasking==0.0.12
namex==0.1.0
numpy==2.2.6
openapi-core==0.19.5
openapi-pydantic==0.5.1
openapi-schema-validator==0.6.3
openapi-spec-validator==0.7.2
opentelemetry-api==1.38.0
opentelemetry-proto==1.38.0
opentelemetry-sdk==1.38.0
opentelemetry-semantic-conventions==0.59b0
opt_einsum==3.4.0
optree==0.17.0
packaging==25.0
pandas==2.3.3
parse==1.20.2
pathable==0.4.4
pathvalidate==3.3.1
patsy==1.0.2
peewee==3.18.2
pillow==12.0.0
platformdirs==4.5.0
protobuf==6.33.0
py-key-value-aio==0.2.8
py-key-value-shared==0.2.8
pyarrow==21.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
pycparser==2.23
pydantic==2.12.3
pydantic-settings==2.11.0
pydantic_core==2.41.4
Pygments==2.19.2
pyparsing==3.2.5
pyperclip==1.11.0
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
python-multipart==0.0.20
pytz==2025.2
pywin32==311
pywin32-ctypes==0.2.3
PyYAML==6.0.3
referencing==0.36.2
requests==2.32.5
rfc3339-validator==0.1.4
rich==14.2.0
rich-rst==1.3.2
rpds-py==0.28.0
rsa==4.9.1
scikit-learn==1.7.2
scipy==1.15.3
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
soupsieve==2.8
SQLAlchemy==2.0.44
sqlparse==0.5.3
sse-starlette==3.0.2
starlette==0.48.0
statsmodels==0.14.5
ta==0.11.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.1.0
threadpoolctl==3.6.0
tomli==2.3.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.38.0
waitress==3.0.2
websockets==15.0.1
Werkzeug==3.1.1
wrapt==2.0.0
yfinance==0.2.66
zipp==3.23.0
