from flask import Flask, render_template
from load_data import get_data_summary, get_duplicate_count
from preprocessing import run_preprocessing
from eda import get_eda_summary_data


app = Flask(__name__)

@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )

@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    duplicate_count = 0

    try:
        summary = get_data_summary()
        duplicate_count = get_duplicate_count()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        Duplicate_count=duplicate_count,
        error=error
    )

@app.route("/preprocessing")
def preprocessing_route():
    error = None
    summary = None

    try:
        summary = run_preprocessing()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        summary=summary,
        error=error
    )

@app.route("/eda")
def eda_route():

    error = None
    summary = None

    try:
        summary = get_eda_summary_data()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        summary=summary,
        error=error
    )



if __name__ == "__main__":
    app.run(debug=True)
