
import tqdm

class SilentTqdm:
    """
    A dummy implementation of tqdm that silences progress bars.
    Handles common tqdm usage patterns:
    - Iterable wrapper: tqdm(iterable)
    - Manual updates: pbar = tqdm(total=...); pbar.update()
    - Context manager: with tqdm(...) as pbar:
    """
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable or []
        
    def __iter__(self):
        return iter(self.iterable)
        
    def update(self, *args, **kwargs):
        pass
        
    def close(self):
        pass
        
    def set_description(self, *args, **kwargs):
        pass
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass

def apply():
    """
    Applies the monkeypatch to tqdm.tqdm.
    Call this before importing any modules that use tqdm.
    """
    tqdm.tqdm = SilentTqdm
