
from .art_gen.artbot import Artbot
from .text_gen.lollms_text_gen import Lollms_Text_Gen
from .text_gen.lollms_text_save import Lollms_Text_Saver
from .video_gen.randomize_video import RandomizeVideo

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Artbot": Artbot,
    "RandomizeVideo":RandomizeVideo,
    "Lollms_Text_Gen": Lollms_Text_Gen,
    "Lollms_Text_Saver": Lollms_Text_Saver
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Artbot": "Artbot Node",
    "Lollms": "Lollms_Text_Gen",
    "Lollms": "Lollms_Text_Saver",
    "RandomizeVideo": "RandomizeVideo"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']