from ..package_manager import PackageManager
if not PackageManager.check_package_installed("lollms_client"):
    PackageManager.install_package("lollms_client")

import comfy.model_management
from lollms_client import LollmsClient
import torch

MAX_RESOLUTION=16384

class Lollms_Text_Gen:
    """
    A Lollms_Text_Gen node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For Lollms_Text_Gen, if `FUNCTION = "execute"` then it will run Lollms_Text_Gen().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an Lollms_Text_Gen.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For Lollms_Text_Gen, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
        self.lollms = LollmsClient()
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "lollms_host": ("STRING",{
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "http://localhost:9600"
                }),
                "prompt": ("STRING",{
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
                "data": ("STRING"),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Response",)

    FUNCTION = "build_prompt"

    #OUTPUT_NODE = False

    CATEGORY = "Lollms/Lollms_Text_Gen"

    def build_prompt(self, clip, lollms_host, build_negative_prompt, width, height, batch_size, prompt):
        #do some processing on the image, in this Lollms_Text_Gen I just invert it
        full_prompt = "!@>system: You are a helpful AI agent. Help the user perform his tasks.\n!@>user:" + prompt + "!@>Lollms_Text_Gen:"
        answer = self.lollms.generate_text(lollms_host,full_prompt)
        return (answer,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"
