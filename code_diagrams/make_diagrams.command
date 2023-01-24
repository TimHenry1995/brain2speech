pyreverse -o png neural_networks.module_converter neural_networks.neural_networks neural_networks.stationary_modules neural_networks.streamable_modules  --filter-mode ALL --module-names y --output-directory "code_diagrams/neural_networks"

pyreverse -o png nodes.fit nodes.load_from_file nodes.monitor nodes.predict --filter-mode ALL --module-names y --output-directory "code_diagrams/nodes"