# ComfyUI_AdvancedRefluxControl

As many of you might noticed, the recently released Reflux model is rather a model for generating multiple variants of an image, but it does not allow for changing an image based on a prompt.

If you use Reflux with an Image and add a prompt, your prompt is just ignored. In general, there is no strength slider or anything in Reflux to control how much the coniditioning image should determine the final outcome of your image.

For this purpose I wrote this little custom node that allows you change the strength of the Reflux effect.

## Examples

I used the following pexel image as an example conditioning image: [[https://www.pexels.com/de-de/foto/29455324/]]

![original](https://github.com/user-attachments/assets/16c8bce5-8eb3-4acf-93e9-847a81e969e0)

Lets say we want to have a similar image, but as comic/cartoon. The prompt I use is "comic, cartoon, vintage comic"

Using Reflux on Flux1-dev I obtain the following image.

**original Reflux setting**
![ComfyUI_00106_](https://github.com/user-attachments/assets/0c5506ef-5131-4b57-962c-ab3703881363)

As you can see, the prompt is vastly ignored. Using the custom node and "medium" setting I obtain

**Reflux medium strength**
![image](https://github.com/user-attachments/assets/eb81a55a-6bdd-43ef-a8da-8d27f210c116)

Lets do the same with anime. The prompt is "anime drawing in anime style. Studio Ghibli, Makoto Shinkai."

As the anime keyword has a strong effect in Flux, we see a better prompt following on default than with comics.

**original Reflux setting**
![image](https://github.com/user-attachments/assets/e5795369-2b8e-477a-974f-e0250d8689b6)

Still, its far from perfect. With "medium" setting we get an image that is much closer to anime or studio Ghibli.

**Reflux medium strength**
![image](https://github.com/user-attachments/assets/b632457a-3a7e-4d99-981e-6c2682d16e2e)


You can also mix more than one images together. Here is an example with adding a second image: [[https://www.pexels.com/de-de/foto/komplizierte-bogen-der-mogul-architektur-in-jaipur-29406307/]]

Mixing both together and using the anime prompt above gives me

![image](https://github.com/user-attachments/assets/1385b22f-4497-4fdf-8255-3a15bda74a1d)

Finally, we try a very challenging prompt: "Marble statues, sculptures, stone statues. stone and marble texture. Two sculptures made out of marble stone.". As you can see, I repeated the prompt multiple times to increase its strength.
But despite the repeats, the default Reflux workflow will just give us the input image refluxed - our prompt is totally ignored.

**original Reflux setting**
![ComfyUI_00108_](https://github.com/user-attachments/assets/24ad66e9-4f21-497d-8d0e-cb4778f0d1e9)

With medium we get an image back that looks more like porcelain instead of marble, but at least the two women are sculptures now.

**Reflux medium strength**
![image](https://github.com/user-attachments/assets/dce4aa6f-52ab-4ef0-b027-193318895969)

Further decreasing the Reflux strength will transform the woman into statues finally, but it will also further decrease their likeness to the conditioning image. In almost all my experiments, it was better to repeat multiple seeds with the "medium" setting instead of further decreasing the strength.

## Usage

You can use the images above for example workflows.

Since last update I added a new node which I would recommend over the old workflow above. Use this image as an example workflow:

![image](https://github.com/user-attachments/assets/b6ee8e4e-2599-499d-9dd4-7fbdc5879e90)

There are two parameters you can play around with:
- downsampling_factor: as larger the value as less information you get from your image. Use a value between 1 and 4 with 1 is the original reflux method. For all example images I used a downsampling_factor of 3, which usually works best. The image above, however, uses a downsampling_factor of 2 and, as you can see, is even closer to the original image while still having a clear anime style
- mode: the method used for downsampling. "area" or "bicubic" work best.

The ComfyUI plugin comes with two additional nodes: StyleModelApplySimple and StyleModelApplyAdvanced. Usually, you can just replace your ApplyStyle node with the StyleModelApplySimple node and "medium" strength and you will get best results. However, feel free to experiment with the StyleModelApplyAdvanced node.

## Short background on Reflux

Reflux works in two steps. First there is a Clip Vision model that crops your input image into square aspect ratio and reduce its size to 384x384 pixels. It splits this image into 27x27 small patches and each patch is projected into CLIP space.

Reflux itself is just a very small linear function that projects these clip image patches into the T5 latent space. The resulting tokens are then added to your T5 prompt.

Intuitively, Reflux is translating your conditioning input image into "a prompt" that is added at the end of your own prompt.

So why is Reflux dominating the final prompt? It's because the user prompt is usually very short (255 or 512 tokens). Reflux, in contrast, adds 729 new tokens to your prompt. This might be 3 times as much as your original prompt. Also, the Reflux prompt might contain much more information than a user written prompt that just contains the word "anime". 

So there are two solutions here: Either we shrink the strength of the Reflux prompt, or we shorten the Reflux prompt.

The next sections are a bit chaotic: I changed the method several times and many stuff I tried is outdated already. The only and best technique I found so far is described in **Interpolation methods**.

## Controling Reflux with Token downsampling
To shrink the Reflux prompt and increase the influence of the user prompt, we can use a simple trick: We take the 27x27 image patches and split them into 9x9 blocks, each containing 3x3 patches. We then merge all 3x3 tokens into one by averaging their latent embeddings. So instead of having a very long prompt with 27x27=729 tokens we now only have 9x9=81 tokens. So our newly added prompt is much smaller than the user provided prompt and, thus, have less influence on the image generation.

Downsampling is what happens when you use the "medium" setting. Of all three techniques I tried to decrease the Reflux effect, downsampling worked best. ~~However, there are no further customization options. You can only downsample to 81 tokens (downsampling more is too much)~~.

## Interpolation methods

Instead of averaging over small blocks of tokens, we can use a convolution function to shrink our 27x27 images patches to an arbitrary size. There are different functions available which most of you probably know from image resizing (its the same procedure). The averaging method above is "area", but there are also other methods available such as "bicubic".  

## Controling Reflux with Token merging

The idea here is to shrink the Reflux prompt length by merging similar tokens together. Just think about large part of your input image contain more or less the same stuff anyways, so why having always 729 tokens? My implementation here is extremely simple and stupid and not very efficient, but anyways: I just go over all Reflux tokens and merge two tokens if their cosine similarity is above a user defined threshold.

Even a threshold like 0.9 is already removing half of the Reflux tokens. A threshold of 0.8 is often reducing the Reflux tokens so much that they are in similar length as the user prompt.

I would start with a threshold of 0.8. If the image is blurry, increase the value a bit. If there is no effect of your prompt, decrease the threshold slightly.

## Controling Reflux with Token downscaling

We can also just multiply the tokens by a certain strength value. As lower the strength, as closer the values are to zero. This is similar to prompt weighting which was quite popular for earlier stable diffusion versions, but never really worked that well for T5. Nevertheless, this technique seem to work well enough for flux.

If you use downscaling, you have to use a very low weight. You can directly start with 0.3 and go down to 0.1 if you want to improve the effect. High weights like 0.6 usually have no impact.

## Doing both or all three?

My feeling currently is that downsampling by far works best. So I would first try downsampling with 1:3 and only use the other options if the effect is too weak or too strong.
