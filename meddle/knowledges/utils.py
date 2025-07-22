from glob import glob
import os.path as osp

class KnowledgeManager:
    def __init__(self):
        self.knowledge_base = {}

    def add_knowledge(self, key, value):
        """Add knowledge to the knowledge base."""
        self.knowledge_base[key] = value

    def add_knowledge_from_file(self, key, file_path):
        """Add knowledge to the knowledge base from a file."""
        try:
            if file_path.endswith(('.txt', '.md', '.markdown')):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                self.knowledge_base[key] = content
            else:
                raise ValueError("Unsupported file type. Only .txt and .md files are allowed.")
        except Exception as e:
            print(f"Error adding knowledge from file: {e}")

    def get_knowledge(self, key):
        """Retrieve knowledge from the knowledge base."""
        return self.knowledge_base.get(key, None)

    def remove_knowledge(self, key):
        """Remove knowledge from the knowledge base."""
        if key in self.knowledge_base:
            del self.knowledge_base[key]

knowledge_manager = KnowledgeManager()
knowledge_root_dir = osp.dirname(__file__)
knowledge_files = glob(osp.join(knowledge_root_dir, "**", "*.md"))
for fname in knowledge_files:
    knowledge_manager.add_knowledge_from_file(osp.basename(fname).replace(".md", ""), fname)

if __name__ == "__main__":
    print(knowledge_manager.get_knowledge("hpo_with_optuna"), end="\n\n\n----\n")
    print(knowledge_manager.get_knowledge("sth_not_existed"), end="\n\n\n----\n")
    print(knowledge_manager.get_knowledge("wsi_tricks"), end="\n\n\n----\n")