class WorkingMemory:
    def __init__(self):
        self.entries = []
        self.hot_context = None

    def capture(self, entry):
        self.entries.append(entry)

    def get_pending(self):
        # Assuming pending entries are those that are not consolidated
        return [entry for entry in self.entries if not entry['consolidated']]

    def mark_consolidated(self, entry_id):
        for entry in self.entries:
            if entry['id'] == entry_id:
                entry['consolidated'] = True
                break

    def expire_old_entries(self):
        # Placeholder for implementation: expiring logic based on entry age
        self.entries = [entry for entry in self.entries if not entry['expired']]

    def get_hot_context(self):
        return self.hot_context

    def soft_forget(self, entry_id):
        self.entries = [entry for entry in self.entries if entry['id'] != entry_id]
