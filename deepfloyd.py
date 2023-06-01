import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class DeepFloyd(object):
    def __init__(self, prompt):
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16",
                                                       torch_dtype=torch.float16)
        self.prompt = prompt

    def generate_single_image(self):
        self.pipe.to("cuda")
        image = self.pipe(self.prompt).images[0]

        # you can save the image with
        image.save(f"doctor_selfie.png")

        '''
        If at some point you get a black image, it may be because the content filter 
        built inside the model might have detected an NSFW result. If you believe this
        shouldn't be the case, try tweaking your prompt or using a different seed.
        '''

    def image_grid(self, imgs, rows, cols):

        # generate several images of the same prompt at once
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    def generate_multiple_images(self, num_images=1):
        # generate multiple images for the same prompt by simply using a list with the
        # same prompt repeated several times. We'll send the list to the pipeline
        # instead of the string we used before.

        multi_prompt = [self.prompt] * num_images
        images = self.pipe(multi_prompt).images
        grid = self.image_grid(images, rows=1, cols=3)

        # you can save the grid with
        grid.save(f"astronaut_rides_horse.png")


if __name__ == '__main__':
    prompt = 'a photograph of a doctor taking a selfie'
    # prompt = input("Enter your prompt: ")
    num_images = 3

    deepf = DeepFloyd(prompt)
    deepf.generate_single_image()
    deepf.generate_multiple_images(num_images)


