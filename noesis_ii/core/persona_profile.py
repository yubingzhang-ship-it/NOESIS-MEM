from dataclasses import dataclass

@dataclass
class MemoryTrace:
    memory: str
    timestamp: str

@dataclass
class Persona:
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

class PersonaProfile:
    def __init__(self):
        self.experiences = []
        self.traces = []
        self.current_persona = None

    def store_experience(self, experience: str):
        # Store experience into memories
        trace = MemoryTrace(memory=experience, timestamp='2026-05-05 05:21:33')
        self.experiences.append(trace)

    def retrieve_by_conditions(self, condition):
        # Retrieve experiences based on a given condition
        return [trace for trace in self.experiences if condition(trace)]

    def get_current_persona(self):
        # Get the current persona profile
        return self.current_persona

    def soft_forget(self, trace: MemoryTrace):
        # Softly forget a memory trace
        if trace in self.experiences:
            self.experiences.remove(trace)

    def create_trace_link(self, trace: MemoryTrace):
        # Create a link for a trace
        self.traces.append(trace)

    def recover_trace(self, trace: MemoryTrace):
        # Recover a forgotten or lost trace
        self.experiences.append(trace)