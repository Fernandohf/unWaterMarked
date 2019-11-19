# UnWaterMarked

Deep learning network to remove watermark from images, based on the paper [On the Effectiveness of Visible Watermarks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Dekel_On_the_Effectiveness_CVPR_2017_paper.pdf).

## Problem Description

A watermarked image, $J$, can be described by superposing a watermark, $W$, to an natural image, $I$. Defining $p=(x,y)$ as the location in pixels and $\alpha(p)$ as a varying opacity, we have that

$$J(p) = \alpha(p)W(p) + (1 - \alpha(p))I(p) $$

In general, watermarks are translucent to maintain certain visibility of the underlying image, which implies $\alpha < 1$.

Following this equation, the inverse process could be described by

$$I(p) = \frac{J(p) - \alpha(p)W(p)}{1 - \alpha(p)}$$

Per pixel, this results in an intrinsic unknown problem with three variable ($W$, $\alpha$, $I$) and only one equation.

However, the consistency on the watermarks could be exploit. Assuming the same $W$ and $\alpha$ are applied to a collections of $K$ images $I_k$. The problem could be restate as shown

$$J_k(p) = \alpha(p)W(p) + (1 - \alpha(p))I_k(p), \qquad k=1,2,...K $$

This becomes a multi-image matting problem that is still under-determined with $3(K+1) + 1$ variable and $3K$ equations per pixel.

### Watermark Estimation and Detection

Assuming the same watermark is applied to the collection of images, it could be estimated by calculating the median of the watermarked images gradients



## Project Organization

------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
