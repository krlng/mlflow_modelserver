# PoC Model-Registry

Dieses Repo enthält [Demo-Code](
https://github.com/krlng/mlflow_modelserver/blob/master/mlflow.ipynb) zum speichern und laden von Modellen in einer [ML-Flow model registry](https://www.mlflow.org/docs/latest/model-registry.html).

Untersuchungsgegenstand:

 * Nutzen einer Scikit-Learn Pipeline mit eigenen Komponenten
 * Model-Serving via Fast-API

Bewusste Einschränkungen:

 * Als Artifact Store wird nur das lokale Dateisystem genutzt (hier wäre ein S3 Bucket o.ä. zum empfehlen)
