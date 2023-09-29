from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
import openai

class ImageDescriptor:

    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.model.to(device)

    def get_image_description(self, image: Image.Image) -> str:
        inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
        description = self._generate(inputs)
        return description

    def _generate(self, inputs) -> str:
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    def ask_question(self, image, question):
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        answer = self._generate(inputs)
        return answer

    def get_intersting_things(self, image):
        question = f"What are three interesting things I can interact with in this image?"
        answer = self.ask_question(image, question)
        return answer

    def investigate_image(self, image: Image.Image, story: str):
        interesting_things = self.get_intersting_things(image)
        objects = interesting_things.split(',')
        clues = [self.get_clue(image, obj, story) for obj in objects]

        answer = f"I found few interesting things: {interesting_things}. \n" + '\n\n'.join(clues)
        return answer

    def get_clue(self, image: Image.Image, object: str, story: str) -> str:
        object_description = self.ask_question(image, f"What is interesting about {object} in the image?")
        image_description = self.get_image_description(image)
        prompt = f"""I am writing a storyline for my game based on the image I have.
        The storyline begins with text below:
        {story}
        Image description is below:
        {image_description} 
        There is an interesting object on the image, object description below:
        {object_description}
        
        I want you write an description of the object in artistic way. Describe what make it object special.
        Assume that there is some detective story behind this object. Add a clue for the player related to this object.
        Your answer should be up to 40 words.
        """
        messages = [{"role": "user", "content": prompt}]
        answer = self.get_answer(messages)
        return answer

    @staticmethod
    def get_answer(messages):
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages,
        )
        return response["choices"][0]["message"]['content']


if __name__ == "__main__":
    image_path = 'data/interior_style.png'
    image = Image.open(image_path).convert('RGB')
    descriptor = ImageDescriptor('cuda:0')
    descriptor.investigate_image(image, 'You walk into the old basement. It looks like droid laboratory.')
