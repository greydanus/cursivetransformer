# Cursive Transformer

_Note (July 5, 2024): this repo is under active development and thus subject to rapid breaking changes._

## Making a dataset

Let's construct a dataset of cursive pen strokes for training a handwriting model. I don't have an e-pen or any special hardware. Also, someday I want to allow people to clone their own handwriting in a demo. Thus we will use a strictly trackpad/mouse-based interface. This interface is defined in the self-contained `collect.html` which is a simple webpage that allows users to enter handwriting samples. It can prompt them with words from a word bank if desired. When they are finished entering samples, they can export the result to a JSON file. I experimented with a couple different approaches to dataset generation (tracing from pictures of cursive, writing multiple words at once, writing single words and then later stitching them together...) so this interface supports them all.

![collect](static/collect.png)


## Preprocessing and Tokenization

Our raw dataset consists of a large JSON file containing many examples of handwriting. Each example is a JSON dictionary containing a 'points' variable that is a list of (`x`, `y`, `is_pen_down`) tuples. This dictionary also has a 'metadata' section which includes the ASCII annotations, the author name, and some other information. Our first preprocessing step is to convert the pen stroke coordinates from absolute coordinates to offsets. This is a much better representation for training a generative model. Next, we compute unit vectors and magnitudes for each of these offsets, so as to decouple magnitude from direction. This permits a more compact tokenization, and is also a better representation for training a generative model. Now our stroke offsets are represented by 4-tuples of (`unit_dx`, `unit_dy`, `magnitude`, and `is_pen_down`).

**Custom tokenizer.** The next step is to tokenize these offsets. We do this by introducing separate lookup tables for `unit_dx`, `unit_dy`, and _a combination of `magnitude` + `is_pen_down`_. Thus token numbers 0-N1 indicate a particular value for `unit_dx`, token numbers N1-N2 represent a particular value for `unit_dy`, and token numbers N2-N3 represent a particular configuration of `magnitude` and `is_pen_down`. Thus, a tuple of three tokens, eg (35, 106, 142) is sufficient for representing one stroke offset. This means, for example, that a handwriting sample of 250 points could be converted to 250 * 3 = 750 tokens. We need 30-40 pen strokes to represent one character, so an example containing 18 characters like the one below ("actor mass afford zoo") would need about 35 * 18 = 630 stroke offsets and 630 * 3 = 1890 tokens. What we below is that this particular example required 537 strokes (downsampled from 1082) which would have corresponded to 1611 tokens. The token sequence was padded out to 3000 tokens, which is why "Encoded stroke shape" is 3000.

This tokenization scheme is a little messy, but one nice feature is that the tokens end up looking just like the tokens that one would end up using for a language modeling task. Thus we are able to use boilerplate LLM Transformer training code from this point onwards.

![tokenizer](static/tokenizer.png)

## Training and logging

We use a Transformer architecture taken from Karpathy's [`makemore`](https://github.com/karpathy/makemore/blob/master/makemore.py) repo. This, in turn, is basically the GPT-2/GPT-3 architecture plus a few simplifications (eg, slightly different GELU, pre-attention layer normalization). We add cross-attention to this architecture in order to condition on the ASCII context information. As for training code, we again start from the [`makemore`](https://github.com/karpathy/makemore/blob/master/makemore.py) repo, but add Weights and Biases for logging and sample visualization.

**A quick ode to Weights and Biases.** One of the challenges of debugging this kind of model is that you need to look at the samples pretty frequently in order to determine what the model is learning and what failure modes are occurring. I've really enjoyed using W&B for this because visualizing samples (as images) while training in the same notebook is not trivial. You need to run two separate threads, and on top of that it's just not what notebooks are designed for. By comparison, W&B makes this easy and extremely fast. I've noticed that images of sample handwriting are ready to view on W&B within a second or two of when they are generated on my A100 Colab GPU. That's pretty impressive! W&B also makes it easy to run ablation studies, as loss stats from different runs (but the same project) are by defauly aggregated in the same plot. This has been of great practical use when doing mini-experiments to determine how to set data augmentation parameters and other modeling parameters. I run four models on four different A100s (one ablatiion setting on each) and compare their stats on W&B in real time.

![wandb](static/wandb.png)


## Samples

_This section is a work in progress._ The samples shown here were generated by conditioning on ASCII characters from the test set examples. They provide some indication that the ASCII information is able to effectively condition what the model generates via cross-attention. You'll notice that the model garbles some characters and often generates them out of order. It has not yet truly solved the task. However, I'm pleased to have gotten this far with a train/test set of 330/36 examples (!) and some data augmentation.

![faith-tall](static/faith-tall.png)

![today](static/today.png)