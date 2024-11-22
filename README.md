# ComfyUI_AdvancedRefluxControl

As many of you might noticed, the recently released Reflux model is rather a model for generating multiple variants of an image, but it does not allow for changing an image based on a prompt.

If you use Reflux with an Image and add a prompt, your prompt is just ignored. In general, there is no strength slider or anything in Reflux to control how much the coniditioning image should determine the final outcome of your image.

For this purpose I wrote this little custom node. It adds two parameters that control the strength of a Reflux. 

## Short background on Reflux

Reflux works in two steps. First there is a Clip Vision model that crops your input image into square aspect ratio and reduce its size to 384x384 pixels. It splits this image into 27x27 small patches and each patch is projected into CLIP space.

Reflux itself is just a very small linear function that projects these clip image patches into the T5 latent space. The resulting tokens are then added to your T5 prompt.

Intuitively, Reflux is translating your conditioning input image into "a prompt" that is added at the end of your own prompt.

So why is Reflux dominating the final prompt? It's because the user prompt is usually very short (255 or 512 tokens). Reflux, in contrast, adds 729 new tokens to your prompt. This might be 4 times as much as your original prompt. Also, the Reflux prompt might contain much more information than a user written prompt that just contains the word "anime". 

So there are two solutions here: Either we shrink the strength of the Reflux prompt, or we shorten the Reflux prompt.

## Controling Reflux with Token merging

The idea here is to shrink the Reflux prompt length by merging similar tokens together. Just think about large part of your input image contain more or less the same stuff anyways, so why having always 729 tokens? My implementation here is extremely simple and stupid and not very efficient, but anyways: I just go over all Reflux tokens and merge two tokens if their cosine similarity is above a user defined threshold.

Even a threshold like 0.9 is already removing half of the Reflux tokens. A threshold of 0.8 is often reducing the Reflux tokens so much that they are in similar length as the user prompt.

I would start with a threshold of 0.8. If the image is blurry, increase the value a bit. If there is no effect of your prompt, decrease the threshold slightly.

## Token merging by downsampling

Instead of merging similar tokens, we can also merge neighboured tokens. Sometimes this seems to harm the outcoming image less, but there are not many possible values: either we downsample by factor 3 or by factor 9. Latter is usually too strong. But I often get good results with factor 3.

## Controling Reflux with Token downscaling

We can also just multiply the tokens by a certain strength value. As lower the strength, as closer the values are to zero. This is similar to prompt weighting which was quite popular for earlier stable diffusion versions, but never really worked that well for T5. Nevertheless, this technique seem to work well enough for flux.

If you use downscaling, you have to use a very low weight. You can directly start with 0.3 and go down to 0.1 if you want to improve the effect. High weights like 0.6 usually have no impact.

## Doing both or all three?

Of course all these techniques can be combined. For me its still unclear what is better. My feeling so far is that downsampling or token merging works better, though, both together is often too much. If you use too low threshold for merging, the image often gets blurry, probably because the tokens have some implicit positional information that gets mixed up during merging. Downsampling seem to not have this issue, but is quite strong in its effect. If token merging or downsampling is not strong enough, you can add token downscaling to further improve the effect.