import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import gc


class DeepFloyd(object):
    def __init__(self, prompt, num_images=10):
        self.pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16",
                                                            torch_dtype=torch.float16)
        self.pipe.to("cuda")
        self.prompt = prompt
        self.num_images = num_images

    def generate_single_image(self):
        image = self.pipe(self.prompt).images[0]

        '''
        If at some point you get a black image, it may be because the content filter 
        built inside the model might have detected an NSFW result. 
        If you believe this shouldn't be the case, try tweaking your prompt or 
        using a different seed.
        '''
        return image

    def save_image(self, image_obj, file_name=""):
        if not file_name:
            file_name = self.prompt.replace(" ", "_")
        if type(image_obj) is list:
            for i, img in enumerate(image_obj):
                img.save("{file_name}.png".format(file_name="{file_name}_{index}".format
                (file_name=file_name, index=i)))

        else:
            image_obj.save("{file_name}.png".format(file_name=file_name))

    def image_grid(self, imgs, rows, cols):
        # generate several images of the same prompt at once
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))

        return grid

    def generate_multiple_images(self):
        images = []

        for i in range(self.num_images):
            image = self.generate_single_image()
            images.append(image)

        return images

    def delete(self):
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    prompt = 'a photograph of a doctor taking a selfie'
    # prompt = input("Enter your prompt: ")
    num_images = 3

    deepf = DeepFloyd(prompt, num_images)
    single_img = deepf.generate_single_image()
    deepf.save_image(single_img)

    multi_img = deepf.generate_multiple_images()
    deepf.save_image(multi_img)

    deepf.delete()


