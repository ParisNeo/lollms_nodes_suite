from ..package_manager import PackageManager
if not PackageManager.check_package_installed("lollms_client"):
    PackageManager.install_package("lollms_client")

import comfy.model_management
from lollms_client import generate_text
import torch
from torchvision.transforms import Compose, Resize, CenterCrop

MAX_RESOLUTION=16384

class Artbot:
    """
    A Artbot node

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
        The name of the entry-point method. For Artbot, if `FUNCTION = "execute"` then it will run Artbot().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an Artbot.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For Artbot, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    
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
                "clip": ("CLIP",),
                "lollms_host": ("STRING",{
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "http://localhost:9600"
                }),
                "build_negative_prompt":(["YES","NO","USE_DEFAULT"],),
                "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "prompt": ("STRING",{
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
            "optional": {
                "input_image":("IMAGE",), 
                "vae":("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("Positive", "Negative", "latent",)

    FUNCTION = "build_prompt"

    #OUTPUT_NODE = False

    CATEGORY = "Lollms/Artbot"

    def build_prompt(self, clip, lollms_host, build_negative_prompt, width, height, batch_size, prompt, input_image=None, vae=None):
        #do some processing on the image, in this Artbot I just invert it
        full_prompt = "!@>system: Act as Artbot, Use the user prompt as a subject then build an image generation prompt for a captivating art.Start by a very simple description of the artwork, then follow up with tags or art styles, here are some examples of tags 'whimsical pop-surrealist style, autumn forest, magical fairies, vibrant colors, highres, 8k, cyberpunk, steampunk, Best quality, UHD, HDR, contemporary impressionism etc', you can also give an information about the camera and the shot parameters if needed. Use as much tags as you need. Only use tags that serve the project of artwork. If needed evoke the name of an artist  This concise prompt sparks curiosity and enriches user's artistic experience.\n!@>user:" + prompt + "\n!@>artbot:"
        positive_prompt = generate_text(lollms_host,full_prompt)
        tokens = clip.tokenize(positive_prompt)
        positive_cond, positive_pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        print(f"Positive conditionning: {positive_cond}")

        if build_negative_prompt=="USE_DEFAULT":
            neg_prompt = "(((ugly))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), ((extra arms)), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), ((watermark)), ((robot eyes))"
        elif build_negative_prompt=="YES":
            full_prompt = "!@>system: Build a list of expressions that shouldn't be in the an artwork built from the user prompt. example (((ugly))), (((duplicate))), ((morbid)), ((mutilated)), out of frame, extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), ((extra arms)), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), ((watermark)), ((robot eyes)).\nUse the user prompt as a base to determine this list and answer only with the list.\n!@>user:" + prompt + "!@>artbot:"
            neg_prompt = generate_text(lollms_host,full_prompt)
        else:
            neg_prompt = ""
        tokens = clip.tokenize(neg_prompt)
        negative_cond, negative_pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        print(f"Negative conditionning: {neg_prompt}")

        if input_image is not None:
            # First, determine how to resize the image without changing its aspect ratio
            def resize_keep_aspect_ratio(img, target_width, target_height):
                # Convert from HWC to CHW format which is expected by torchvision transforms
                img = img.permute(2, 0, 1)
                # Get the original dimensions
                orig_height, orig_width = img.shape[1], img.shape[2]
                # Calculate the scaling factors for both dimensions
                scale_width = target_width / orig_width
                scale_height = target_height / orig_height
                # Choose the larger scaling factor to ensure the image covers the target dimensions
                scale_factor = min(scale_width, scale_height)
                # Calculate the resized dimensions
                new_width = int(orig_width * scale_factor)
                new_height = int(orig_height * scale_factor)
                # Resize the image
                resize_transform = Resize((new_height, new_width))
                resized_img = resize_transform(img)

                return resized_img
            # Apply the transformations to each image in the batch
            processed_images = []
            for i in range(input_image.shape[0]):
                img = input_image[i,...]
                resized_img = resize_keep_aspect_ratio(img, width, height)
                # Now, center crop to the target dimensions
                crop_transform = CenterCrop((height, width))
                cropped_img = crop_transform(resized_img)
                # Convert from HWC to CHW format which is expected by torchvision transforms
                cropped_img = cropped_img.permute(1, 2, 0)
                
                processed_images.append(cropped_img)

            # Stack processed images back into a batch
            processed_batch = torch.stack(processed_images*batch_size)
            # Encode the processed batch using VAE
            latent = vae.encode(processed_batch)
        else:
            latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ([[positive_cond, {"pooled_output": positive_pooled}]], 
                [[negative_cond, {"pooled_output": negative_pooled}]],
                {"samples":latent},)

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

