import argparse
import json
import webbrowser
from threading import Lock, Timer

from flask import Flask, Response, render_template, request, stream_with_context
from session import Session, _sessions_dir, get_initial_session, load_session, new_session, prepare_sessions
from waitress import serve

import model_init
from model import ExLlama
from tokenizer import ExLlamaTokenizer

app = Flask(__name__)
app.static_folder = "static"
generate_lock = Lock()
session: Session


# Render template
@app.route("/")
def home():
    return render_template("index.html")


# Get existing sessions
@app.route("/api/populate")
def api_populate():
    global session
    return session.api_populate()


# Edit block
@app.route("/api/edit_block", methods=["POST"])
def api_edit_block():
    global session
    data = request.get_json()
    session.api_edit_block(data)
    return json.dumps({"result": "ok"}) + "\n"


# Delete block
@app.route("/api/delete_block", methods=["POST"])
def api_delete_block():
    global session
    data = request.get_json()
    session.api_delete_block(data)
    return json.dumps({"result": "ok"}) + "\n"


# Rename session
@app.route("/api/rename_session", methods=["POST"])
def api_rename_session():
    global session
    data = request.get_json()
    success = session.api_rename_session(data)
    return json.dumps({"result": "ok" if success else "fail"}) + "\n"


# Delete session
@app.route("/api/delete_session", methods=["POST"])
def api_delete_session():
    global session
    data = request.get_json()
    session.api_delete_session(data)
    return json.dumps({"result": "ok"}) + "\n"


# Set fixed prompt settings
@app.route("/api/set_fixed_prompt", methods=["POST"])
def api_set_fixed_prompt():
    global session
    data = request.get_json()
    session.api_set_fixed_prompt(data)
    return json.dumps({"result": "ok"}) + "\n"


# Set generation settings
@app.route("/api/set_gen_settings", methods=["POST"])
def api_set_gen_settings():
    global session
    data = request.get_json()
    session.api_set_gen_settings(data)
    return json.dumps({"result": "ok"}) + "\n"


# Set session
@app.route("/api/set_session", methods=["POST"])
def api_set_session():
    global session
    data = request.get_json()
    load_session_name = data["session_name"]
    if load_session_name == ".":
        session = new_session()
    else:
        session = load_session(load_session_name, append_path=True)
    return json.dumps({"result": "ok"}) + "\n"


# Set participants
@app.route("/api/set_participants", methods=["POST"])
def api_set_participants():
    global session
    data = request.get_json()
    session.api_set_participants(data)
    return json.dumps({"result": "ok"}) + "\n"


# Accept input
@app.route("/api/userinput", methods=["POST"])
def api_userinput():
    data = request.get_json()
    user_input = data["user_input"]

    with generate_lock:
        global session
        result = Response(stream_with_context(session.respond_multi(user_input)), mimetype="application/json")
        return result


@app.route("/api/append_block", methods=["POST"])
def api_append_block():
    data = request.get_json()
    session.api_append_block(data)
    return json.dumps({"result": "ok"}) + "\n"


def main(args: argparse.Namespace):
    model_init.get_model_files(args)
    model_init.print_options(args)
    config = model_init.make_config(args)
    print(" -- Loading model...")
    model = ExLlama(config)
    print(" -- Loading tokenizer...")
    tokenizer = ExLlamaTokenizer(args.tokenizer)

    model_init.print_stats(model)

    # Get the session ready
    prepare_sessions(model, tokenizer, args.sessions_dir)
    global session
    session = get_initial_session()
    print(f" -- Sessions stored in: {_sessions_dir()}")

    # Start the web server
    machine = args.host
    host, port = machine.split(":")

    if host == "localhost":
        Timer(1, lambda: webbrowser.open(f"http://{machine}/")).start()

    serve(app, host=host, port=port)


if __name__ == "__main__":
    # Load the model
    parser = argparse.ArgumentParser(description="Simple web-based chatbot for ExLlama")
    parser.add_argument(
        "-host", "--host", type=str, help="IP:PORT eg, 0.0.0.0:7862", default="localhost:5000"
    )
    parser.add_argument(
        "-sd",
        "--sessions_dir",
        type=str,
        help="Location for storing user sessions, default: ~/exllama_sessions/",
        default="~/exllama_sessions/",
    )

    model_init.add_args(parser)
    args = parser.parse_args()
    model_init.post_parse(args)
    main(args)
