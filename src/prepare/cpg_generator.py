import json
import re
import subprocess
import os.path
import os
import sys
import time
from .cpg_client_wrapper import CPGClientWrapper

# from ..data import datamanager as data


def funcs_to_graphs(funcs_path):
    client = CPGClientWrapper()
    # query the cpg for the dataset
    print(f"Creating CPG.")
    graphs_string = client(funcs_path)
    # removes unnecessary namespace for object references
    graphs_string = re.sub(
        r"io\.shiftleft\.codepropertygraph\.generated\.", "", graphs_string
    )
    graphs_json = json.loads(graphs_string)

    return graphs_json["functions"]


def graph_indexing(graph):
    file_path = graph.get("file", "")
    file_name = file_path.replace("\\", "/").split("/")[-1]
    match = re.search(r"(\d+)\.c$", file_name)
    if not match:
        return None
    idx = int(match.group(1))
    graph = dict(graph)
    graph.pop("file", None)
    return idx, {"functions": [graph]}


def joern_parse(joern_path, input_path, output_path, file_name):
    out_file = file_name + ".bin"

    # Convert to absolute paths to avoid classpath issues
    abs_input_path = os.path.abspath(input_path)
    abs_output_path = os.path.abspath(os.path.join(output_path, out_file))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)

    # On Windows, c2cpg.bat is broken (empty classpath bug)
    # Workaround: call java directly with the launcher JAR
    if sys.platform == "win32":
        c2cpg_dir = os.path.join(joern_path, "frontends", "c2cpg")
        lib_dir = os.path.join(c2cpg_dir, "lib")
        launcher_jar = os.path.join(lib_dir, "io.joern.c2cpg-4.0.473-launcher.jar")

        if not os.path.exists(launcher_jar):
            # Try to find the launcher jar dynamically
            import glob

            launcher_jars = glob.glob(
                os.path.join(lib_dir, "io.joern.c2cpg-*-launcher.jar")
            )
            if launcher_jars:
                launcher_jar = launcher_jars[0]
            else:
                raise FileNotFoundError(
                    f"Could not find c2cpg launcher JAR in {lib_dir}"
                )

        # Build Java command directly
        cmd = f'java -XX:+UseG1GC -Xmx4080m -jar "{launcher_jar}" "{abs_input_path}" --output "{abs_output_path}"'

        joern_parse_call = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    else:
        executable = "joern-parse"
        joern_executable = os.path.join(joern_path, executable)
        cmd = [joern_executable, abs_input_path, "--output", abs_output_path]

        joern_parse_call = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    if joern_parse_call.stdout:
        print(joern_parse_call.stdout)
    if joern_parse_call.stderr:
        print("STDERR:", joern_parse_call.stderr)

    if joern_parse_call.returncode != 0:
        raise subprocess.CalledProcessError(
            joern_parse_call.returncode,
            joern_parse_call.args,
            joern_parse_call.stdout,
            joern_parse_call.stderr,
        )

    return out_file


def joern_create(joern_path, in_path, out_path, cpg_files):
    # On Windows, use .bat extension; on Unix, no extension
    executable = "joern.bat" if sys.platform == "win32" else "joern"
    joern_executable = os.path.join(joern_path, executable)

    json_files = []
    for cpg_file in cpg_files:
        json_file_name = f"{cpg_file.split('.')[0]}.json"
        json_files.append(json_file_name)

        cpg_file_path = os.path.join(in_path, cpg_file)
        print(cpg_file_path)
        if os.path.exists(cpg_file_path):
            # Use forward slashes for Joern (Scala/Java expects them even on Windows)
            json_out = os.path.abspath(os.path.join(out_path, json_file_name)).replace(
                "\\", "/"
            )
            cpg_path = os.path.abspath(cpg_file_path).replace("\\", "/")
            # Script is in joern/ directory at workspace root
            script_path = os.path.abspath(
                os.path.join("joern", "graph-for-funcs.sc")
            ).replace("\\", "/")

            print(f"Importing CPG from: {cpg_path}")
            print(f"Running script: {script_path}")
            print(f"Output to: {json_out}")

            # Create a temporary command file for this CPG
            cmd_file = os.path.join(out_path, f"_temp_{cpg_file.split('.')[0]}.sc")
            with open(cmd_file, "w", encoding="utf-8") as f:
                f.write(f'importCpg("{cpg_path}")\n')
                # Inline the Scala helper script and invoke generator
                with open(script_path, "r", encoding="utf-8") as script:
                    script_content = script.read()
                    f.write(script_content + "\n")
                f.write("val jsonResult = generateGraphs()\n")
                f.write(
                    f'java.nio.file.Files.write(java.nio.file.Paths.get("{json_out}"), jsonResult.getBytes(java.nio.charset.StandardCharsets.UTF_8))\n'
                )
                f.write("delete\n")
                f.write("exit\n")

            # Run joern with the command file
            try:
                result = subprocess.run(
                    [joern_executable, "--script", cmd_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=120,
                )

                if result.stdout:
                    print(f"Joern output:\n{result.stdout}")
                if result.stderr:
                    print(f"Joern errors:\n{result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"WARNING: Joern timed out processing {cpg_file}")
            finally:
                # Clean up temp file
                if os.path.exists(cmd_file):
                    os.remove(cmd_file)

    return json_files


def json_process(in_path, json_file):
    json_file_path = os.path.join(in_path, json_file)
    if os.path.exists(json_file_path):
        with open(json_file_path, encoding="utf-8") as jf:
            cpg_string = jf.read()
            cpg_string = re.sub(
                r"io\.shiftleft\.codepropertygraph\.generated\.", "", cpg_string
            )
            cpg_json = json.loads(cpg_string)
            container = []
            for graph in cpg_json["functions"]:
                if graph.get("file") == "N/A":
                    continue
                indexed = graph_indexing(graph)
                if indexed is not None:
                    container.append(indexed)
            return container
    return None


"""
def generate(dataset, funcs_path):
    dataset_size = len(dataset)
    print("Size: ", dataset_size)
    graphs = funcs_to_graphs(funcs_path[2:])
    print(f"Processing CPG.")
    container = [graph_indexing(graph) for graph in graphs["functions"] if graph["file"] != "N/A"]
    graph_dataset = data.create_with_index(container, ["Index", "cpg"])
    print(f"Dataset processed.")

    return data.inner_join_by_index(dataset, graph_dataset)
"""

# client = CPGClientWrapper()
# client.create_cpg("../../data/joern/")
# joern_parse("../../joern/joern-cli/", "../../data/joern/", "../../joern/joern-cli/", "gen_test")
# print(funcs_to_graphs("/data/joern/"))
"""
while True:
    raw = input("query: ")
    response = client.query(raw)
    print(response)
"""
