import importlib
import sys
import unittest


class AgentImportTests(unittest.TestCase):
    def test_agent_module_imports_without_initializing_rag(self):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")

        self.assertTrue(hasattr(module, "graph"))
        self.assertIsNotNone(module.graph)

    def test_call_model_returns_local_fallback_when_gemini_is_unavailable(self):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        module.document_chunks = ["QKD es un método para intercambiar claves de cifrado mediante física cuántica."]
        module.retriever = None
        module.llm = None
        module.initialization_error = "quota exceeded"
        module.api_key = None

        state = {"messages": [{"content": "¿Qué es QKD?"}]}
        result = module.call_model(state)

        self.assertIn("QKD", result["messages"][0].content)

    def test_keyword_fallback_uses_local_terms(self):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        module.document_chunks = ["QKD es un método para intercambiar claves de cifrado mediante física cuántica."]
        answer = module._keyword_fallback_answer("¿Qué es QKD?", "quota exceeded")

        self.assertIn("QKD", answer)
        self.assertIn("quota exceeded", answer)
        self.assertIn("método", answer)

    def test_fallback_returns_surrounding_sentences(self):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        module.document_chunks = [
            "Paso 1. Reemplazo de una OTU. Para reemplazar una OTU por otra en un dominio: 1 Agregue la nueva OTU: con un nombre diferente al que se va a sustituir. 2 Retire la OTU anterior. 3 Asegúrese de que la nueva OTU esté configurada correctamente."
        ]
        answer = module._keyword_fallback_answer("Replacing an OTU", "quota exceeded")

        self.assertIn("Reemplazo de una OTU", answer)
        self.assertIn("Agregue la nueva OTU", answer)
        self.assertIn("Retire la OTU anterior", answer)

    def test_fallback_skips_table_of_contents_chunks(self):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        module.document_chunks = [
            "Table of Contents\nChapter 1..........1\nChapter 2..........5\nChapter 3..........12\n",
            "Monitoring a link\nOnce OTU is created on ONMSi, the user can assign a monitored fiber to an optical switch port of the OTU. This chapter provides a description on the link monitoring process."
        ]
        answer = module._keyword_fallback_answer("link monitoring OTU", "quota exceeded")

        self.assertIn("Monitoreo de un enlace", answer)
        self.assertIn("puede asignar una fibra monitoreada a un puerto de conmutador óptico", answer)
        self.assertNotIn("Table of Contents", answer)


if __name__ == "__main__":
    unittest.main()
