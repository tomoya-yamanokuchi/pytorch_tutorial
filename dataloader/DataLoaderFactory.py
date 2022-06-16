
from .MNISTDataLoader import MNISTDataLoader


class DataLoaderFactory:
    def create(self, name: str):
        name = name.lower()
        if name == "mnist": return MNISTDataLoader