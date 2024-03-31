
from .conditionning_gen.artbot import Artbot
from .text_gen.lollms_text_gen import Lollms_Text_Gen


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Artbot": Artbot,
    "Lollms_Text_Gen": Lollms_Text_Gen
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Artbot": "Artbot Node",
    "Lollms": "Lollms_Text_Gen"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']