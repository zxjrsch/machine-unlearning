from reporter import *


def test_table():
    config = LaTeXTableGeneratorConfig()
    table_generator = LaTeXTableGenerator(config)
    table_generator.generate_tables()
