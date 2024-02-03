from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningData
from invokeai.backend.stable_diffusion.diffusers_pipeline import ControlNetData, T2IAdapterData
from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import PostprocessingSettings
from invokeai.app.invocations.primitives import LatentsField, ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin
from invokeai.app.invocations.compel import ConditioningField
from pydantic import BaseModel, Field
import torch
from typing import Literal, Optional, Callable, List, Union
from PIL import ImageFilter, Image, ImageDraw, ImageFont
import random
import torch.nn.functional as F
#importing libraries 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from io import BytesIO

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Input,
    InputField,
    OutputField,
    InvocationContext,
    UIType,
    invocation,
    invocation_output,
)

@invocation_output("image_and_mask")
class ImageAndMaskOutput(BaseInvocationOutput):
    image: ImageField = OutputField(
        title="Image",
        description="The output image",
    )
    mask: ImageField = OutputField(
        title="Mask",
        description="The output mask",
    )
    width: int = OutputField(
        title="Width",
        description="The width of the output image",
    )
    height: int = OutputField(
        title="Height",
        description="The height of the output image",
    )
    

@invocation("depth_to_3d",
            title="Depth to 3D",
            tags=["image", "depth", "3D"],
            category="image",
            version="1.2.0",)
class DepthTo3DInvocation(BaseInvocation):
    image: ImageField = InputField(
        title="Base Image",
        description="The image to be modified",
    )
    depth: ImageField = InputField(
        title="Depth Map",
        description="The depth image to be used for 3D reconstruction",
    )
    method: Literal["Stereoscopic", "3D Anaglyph"] = InputField(
        title="Method",
        description="The method to be used for the image output",
        default="Stereoscopic",
    )
    shift_factor: float = InputField(
        title="Shift Factor",
        description="The shift factor to be used for the stereoscopic method",
        default=50.0,
    )
    cross_eye: bool = InputField(
        title="Cross Eye",
        description="Reverse the left and right images for cross-eye viewing",
        default=False,
    )

    def stereoscopic(self, image, depth):
        # Convert the PIL images to cv2 images
        image = np.array(image)
        depth_map = np.array(depth)

        # Convert the depth map to floating point for better precision
        depth_map = depth_map.astype(np.float32)

        # Normalize the depth map to the range [0, 1]
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

        # Calculate the shift for each color channel based on the depth map
        shift_map = (depth_map_normalized - 0.5) * self.shift_factor
        shift_map = shift_map[:, :, np.newaxis]  # Add a new axis for broadcasting

        # Create an empty canvas for the left and right stereoscopic images
        left_stereoscopic_image = np.zeros_like(image)
        right_stereoscopic_image = np.zeros_like(image)

        left_tracking = np.zeros_like(image)
        right_tracking = np.zeros_like(image)

        # Iterate through each pixel and calculate the shifted positions
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                shift = int(shift_map[y, x, 0])
                new_x_left = x + shift
                new_x_right = x - shift

                # Check if the new positions are within bounds
                if 0 <= new_x_left < image.shape[1]:
                    left_stereoscopic_image[y, new_x_left, :] = image[y, x, :]
                    left_tracking[y, new_x_left, :] = 255
                if 0 <= new_x_right < image.shape[1]:
                    right_stereoscopic_image[y, new_x_right, :] = image[y, x, :]
                    right_tracking[y, new_x_right, :] = 255

        # Concatenate left and right stereoscopic images
        if self.cross_eye:
            left_stereoscopic_image, right_stereoscopic_image = right_stereoscopic_image, left_stereoscopic_image
            left_tracking, right_tracking = right_tracking, left_tracking
        stereoscopic_image = np.concatenate((left_stereoscopic_image, right_stereoscopic_image), axis=1)
        tracking_image = np.concatenate((left_tracking, right_tracking), axis=1)
        
        # Convert tracking_image to uint8
        tracking_image_mask = (255 - tracking_image).astype(np.uint8)

        # Convert tracking_image_mask to single-channel image
        tracking_image_mask = cv2.cvtColor(tracking_image_mask, cv2.COLOR_RGB2GRAY)

        filled_stereoscopic_image = cv2.inpaint(stereoscopic_image, tracking_image_mask, 3, cv2.INPAINT_TELEA)

        # Convert the numpy array back to a PIL image
        stereoscopic_image = Image.fromarray(filled_stereoscopic_image)
        tracking_image = Image.fromarray(tracking_image) 


        return stereoscopic_image, tracking_image

    def invoke(self, context: InvocationContext) -> ImageAndMaskOutput:
        # Get the image and depth map
        image = context.services.images.get_pil_image(self.image.image_name)
        depth = context.services.images.get_pil_image(self.depth.image_name)

        # resize depth map to match image
        depth = depth.resize(image.size)

        #convert depth map to grayscale
        depth = depth.convert("L")

        # Convert the depth map to a 3D image
        if self.method == "Stereoscopic":
            output, mask = self.stereoscopic(image, depth)
        else:
            output = image
            mask = depth

        image_dto = context.services.images.create(
            image=output,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            is_intermediate=True,
            session_id=context.graph_execution_state_id,
            workflow=context.workflow,
        )

        mask_dto = context.services.images.create(
            image=mask,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            is_intermediate=True,
            session_id=context.graph_execution_state_id,
            workflow=context.workflow,
        )

        # Return the output
        return ImageAndMaskOutput(
            image=ImageField(image_name=image_dto.image_name),
            mask=ImageField(image_name=mask_dto.image_name),
            width=output.width,
            height=output.height,
        )