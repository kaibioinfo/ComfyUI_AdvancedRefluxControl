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

## Short background on Reflux

Reflux works in two steps. First there is a Clip Vision model that crops your input image into square aspect ratio and reduce its size to 384x384 pixels. It splits this image into 27x27 small patches and each patch is projected into CLIP space.

Reflux itself is just a very small linear function that projects these clip image patches into the T5 latent space. The resulting tokens are then added to your T5 prompt.

Intuitively, Reflux is translating your conditioning input image into "a prompt" that is added at the end of your own prompt.

So why is Reflux dominating the final prompt? It's because the user prompt is usually very short (255 or 512 tokens). Reflux, in contrast, adds 729 new tokens to your prompt. This might be 4 times as much as your original prompt. Also, the Reflux prompt might contain much more information than a user written prompt that just contains the word "anime". 

So there are two solutions here: Either we shrink the strength of the Reflux prompt, or we shorten the Reflux prompt.

## Token merging by downsampling
To shrink the Reflux prompt and increase the influence of the user prompt, we can use a simple trick: Merge neighbouring tokens in Flux.. Sometimes this seems to harm the outcoming image less, but there are not many possible values: either we downsample by factor 3 or by factor 9. Latter is usually too strong. But I often get good results with factor 3.

## Controling Reflux with Token merging

The idea here is to shrink the Reflux prompt length by merging similar tokens together. Just think about large part of your input image contain more or less the same stuff anyways, so why having always 729 tokens? My implementation here is extremely simple and stupid and not very efficient, but anyways: I just go over all Reflux tokens and merge two tokens if their cosine similarity is above a user defined threshold.

Even a threshold like 0.9 is already removing half of the Reflux tokens. A threshold of 0.8 is often reducing the Reflux tokens so much that they are in similar length as the user prompt.

I would start with a threshold of 0.8. If the image is blurry, increase the value a bit. If there is no effect of your prompt, decrease the threshold slightly.

## Controling Reflux with Token downscaling

We can also just multiply the tokens by a certain strength value. As lower the strength, as closer the values are to zero. This is similar to prompt weighting which was quite popular for earlier stable diffusion versions, but never really worked that well for T5. Nevertheless, this technique seem to work well enough for flux.

If you use downscaling, you have to use a very low weight. You can directly start with 0.3 and go down to 0.1 if you want to improve the effect. High weights like 0.6 usually have no impact.

## Doing both or all three?

My feeling currently is that downsampling by far works best. So I would first try downsampling with 1:3 and only use the other options if the effect is too weak or too strong.
